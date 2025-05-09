# stream_video_wifi_detector.py
"""
Encrypted Wi-Fi (802.11) Streaming Video Detector
Supports feature extraction, deep-learning training, live detection, and XGBoost feature importance.
Modes:
  extract     - extract features + labels from monitor-mode .pcap
  train       - train deep learning model or XGBoost on extracted data
  live        - sniff live 802.11 packets and classify suspicious streams
"""
import argparse
import csv
import joblib
import sys

import numpy as np
from scapy.all import PcapReader, sniff, RadioTap, Dot11

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# import XGBoost
from xgboost import XGBClassifier

# ----------------------- Packet & Window Features ----------------------- #


def pkt_features(pkt, prev_ts):
    ts = pkt.time
    inter = ts - prev_ts if prev_ts else 0.0
    length = len(pkt)
    subtype = pkt.subtype if hasattr(pkt, "subtype") else 0
    seqnum = (pkt.SC >> 4) & 0xFFF if hasattr(pkt, "SC") else 0
    sig = (
        pkt.dBm_AntSignal
        if hasattr(pkt, "dBm_AntSignal") and pkt.dBm_AntSignal is not None
        else 0.0
    )
    rate = pkt.Rate if hasattr(pkt, "Rate") and pkt.Rate is not None else 0.0
    retry = 1 if hasattr(pkt, "FCfield") and pkt.FCfield & 0x08 else 0
    mcs_known = 1 if hasattr(pkt, "MCS") else 0
    mcs_idx = pkt.MCS.mcs if mcs_known else 0
    mcs_bw = pkt.MCS.bw if mcs_known else 0
    ant = pkt.Antenna if hasattr(pkt, "Antenna") else 0
    return [
        inter,
        length,
        subtype,
        seqnum,
        sig,
        rate,
        retry,
        mcs_known,
        mcs_idx,
        mcs_bw,
        ant,
    ], ts


def window_stats(samples):
    arr = np.array(samples, dtype=float)
    inter = arr[:, 0]
    length = arr[:, 1]
    sig = arr[:, 4]
    rate = arr[:, 5]
    retry = arr[:, 6]
    mcs_idx = arr[:, 8]
    ant = arr[:, 10]

    feats = []
    for col in (length, inter, sig, rate):
        mean = float(np.mean(col))
        var = float(np.var(col))
        feats += [mean, var]
        std = float(np.std(col)) if np.std(col) > 0 else 1.0
        skew = float(np.mean((col - mean) ** 3) / (std**3))
        kurt = float(np.mean((col - mean) ** 4) / (std**4) - 3)
        feats += [skew, kurt]

    feats += [float(np.max(inter)), float(np.min(inter))]

    med = float(np.median(inter))
    bursts = np.split(inter, np.where(inter > med)[0])
    burst_lens = [len(b) for b in bursts]
    feats += [float(np.max(burst_lens)), float(np.mean(burst_lens))]

    feats.append(float(np.sum(retry)))
    feats += [
        float(np.unique(mcs_idx).size),
        float(np.mean(mcs_idx)),
        float(np.var(mcs_idx)),
    ]
    feats.append(float(np.unique(ant).size))
    return feats


# -------------------------- Dataset ------------------------------- #


class WiFiDataset(Dataset):
    def __init__(self, feature_files, label_files):
        X_list = [np.loadtxt(f, delimiter=",") for f in feature_files]
        y_list = [np.loadtxt(l) for l in label_files]
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------------- Model -------------------------------- #


class WiFiNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------- Commands ----------------------------- #


def cmd_extract(args):
    reader = PcapReader(args.pcap)
    feats_f = open(args.output, "w", newline="")
    writer = csv.writer(feats_f)
    label_f = open(args.labels, "w")
    samples, prev_ts = [], 0.0

    for pkt in reader:
        if not pkt.haslayer(RadioTap) or not pkt.haslayer(Dot11):
            continue
        dot11 = pkt[Dot11]
        if dot11.type != 2:  # Data frame only
            continue
        fts, prev_ts = pkt_features(pkt[RadioTap], prev_ts)
        samples.append(fts)
    reader.close()

    n = args.window
    for i in range(len(samples) - n + 1):
        stat = window_stats(samples[i : i + n])
        writer.writerow(stat)
        label_f.write(f"{args.label}\n")

    feats_f.close()
    label_f.close()
    print(f"Extracted {len(samples) - n + 1} windows")


def cmd_train(args):
    # Load data
    X_list = [np.loadtxt(f, delimiter=",") for f in args.features]
    y_list = [np.loadtxt(l) for l in args.labels]
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # XGBoost branch
    if getattr(args, "use_xgb", False):
        neg, pos = np.bincount(y.astype(int))
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=neg / pos,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_val)
        y_prob = xgb.predict_proba(X_val)[:, 1]
        print("=== XGBoost Validation ===")
        print(confusion_matrix(y_val, y_pred))
        print(classification_report(y_val, y_pred, digits=4))
        print("ROC AUC:", roc_auc_score(y_val, y_prob))

        fi = xgb.feature_importances_
        feat_names = [f"f{i}" for i in range(X.shape[1])]
        imp = sorted(zip(feat_names, fi), key=lambda x: x[1], reverse=True)
        print("\nFeature importances (top 10):")
        for name, score in imp[:10]:
            print(f"  {name:>6}: {score:.4f}")

        joblib.dump(xgb, args.model)
        print(f"\nSaved XGBoost model to {args.model}")
        return

    # Deep Learning branch
    ds = WiFiDataset(args.features, args.labels)
    counts = np.bincount(ds.y.numpy().astype(int))
    class_weights = 1.0 / counts
    sample_weights = class_weights[ds.y.numpy().astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    loader = DataLoader(ds, batch_size=args.batch, sampler=sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WiFiNet(ds.X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(args.epochs):
        model.train()
        total, loss_acc = 0, 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total += yb.size(0)
            loss_acc += loss.item() * yb.size(0)
        print(f"Epoch [{ep+1}/{args.epochs}] Loss: {loss_acc/total:.4f}")

    torch.save(model.state_dict(), args.model)
    print(f"Saved DL model to {args.model}")


def cmd_live(args):
    # Load DL model
    ds = WiFiDataset([args.features[0]], [args.labels[0]])  # dummy to get dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WiFiNet(ds.X.shape[1]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    buf, prev_ts = [], 0.0

    def handle(pkt):
        nonlocal prev_ts
        if not pkt.haslayer(RadioTap) or not pkt.haslayer(Dot11):
            return
        dot11 = pkt[Dot11]
        if dot11.type != 2:
            return
        fts, prev_ts = pkt_features(pkt[RadioTap], prev_ts)
        buf.append(fts)
        if len(buf) > args.window:
            buf.pop(0)
        if len(buf) == args.window:
            stat = window_stats(buf)
            x = torch.tensor(stat, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            if pred == 1:
                print("[ALERT] Suspicious at", pkt.time)

    sniff(iface=args.iface, prn=handle, store=False)


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    # extract
    e = sub.add_parser("extract")
    e.add_argument("-f", "--pcap", required=True)
    e.add_argument("-o", "--output", required=True)
    e.add_argument("-l", "--labels", required=True)
    e.add_argument("-w", "--window", type=int, default=11)
    e.add_argument("--label", type=int, default=1)
    # train
    t = sub.add_parser("train")
    t.add_argument("--features", nargs="+", required=True)
    t.add_argument("--labels", nargs="+", required=True)
    t.add_argument("-m", "--model", default="wifi_dl.pth")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument(
        "--use-xgb",
        action="store_true",
        help="Train & report XGBoost feature importances",
    )
    # live
    l = sub.add_parser("live")
    l.add_argument("-i", "--iface", required=True)
    l.add_argument("-m", "--model", default="wifi_dl.pth")
    l.add_argument("-w", "--window", type=int, default=11)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "live":
        cmd_live(args)
    else:
        print("Unknown command")
