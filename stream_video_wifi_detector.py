#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Encrypted Wi-Fi Streaming Video Detector
Modes:
  extract     - windowed packet features for FFNN/LSTM/Transformer
  train       - train ffnn/lstm/transformer or XGBoost on those features
  anomaly     - One-Class SVM on normal data
  live        - sniff live 802.11 and classify per window
"""

import argparse, csv, joblib, sys
import numpy as np
from collections import defaultdict
from scapy.all import PcapReader, sniff, RadioTap, Dot11, IP, TCP, UDP

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# ---------------------- packet & window features ---------------------- #


def pkt_features(pkt, prev_ts):
    ts = float(pkt.time)
    inter = ts - prev_ts if prev_ts else 0.0
    length = float(len(pkt))
    sig = float(getattr(pkt, "dBm_AntSignal", 0.0) or 0.0)
    rate = float(getattr(pkt, "Rate", 0.0) or 0.0)
    retry = 1 if (hasattr(pkt, "FCfield") and pkt.FCfield & 0x08) else 0
    return [inter, length, sig, rate, retry], ts


# -------------------------- extract command --------------------------- #


def cmd_extract(args):
    reader = PcapReader(args.pcap)
    wf = open(args.output, "w", newline="")
    writer = csv.writer(wf)
    lf = open(args.labels, "w")
    samples, prev_ts = [], 0.0

    for pkt in reader:
        if not pkt.haslayer(RadioTap) or not pkt.haslayer(Dot11):
            continue
        if pkt[Dot11].type != 2:
            continue
        fts, prev_ts = pkt_features(pkt[RadioTap], prev_ts)
        samples.append(fts)
    reader.close()

    W = args.window
    for i in range(len(samples) - W + 1):
        window = samples[i : i + W]
        if args.model_type == "ffnn":
            arr = np.array(window)
            feats = []
            for col in arr.T:
                feats += [float(col.mean()), float(col.var())]
            writer.writerow(feats)
        else:
            flat = np.array(window).reshape(-1).tolist()
            writer.writerow(flat)
        lf.write(f"{args.label}\n")

    wf.close()
    lf.close()
    print(f"Extracted {len(samples)-W+1} windows")


# ------------------------- dataset classes --------------------------- #


class SequenceDataset(Dataset):
    def __init__(self, feat_files, label_files, window, model_type):
        # skiprows=1 to ignore header
        X_list = [np.loadtxt(f, delimiter=",", skiprows=1) for f in feat_files]
        y_list = [np.loadtxt(l) for l in label_files]
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        self.model_type = model_type
        self.W = window

        if model_type == "ffnn":
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            pkt_dim = X.shape[1] // window
            self.X = torch.tensor(X.reshape(-1, window, pkt_dim), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# --------------------------- model classes --------------------------- #


class FFNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


class LSTMNet(nn.Module):
    def __init__(self, pkt_dim, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(pkt_dim, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class TransformerNet(nn.Module):
    def __init__(self, pkt_dim, nhead=4, nhid=64, nlayers=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=pkt_dim, nhead=nhead, dim_feedforward=nhid
        )
        self.trans = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.fc = nn.Sequential(nn.Linear(pkt_dim, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        x = x.permute(1, 0, 2)  # [W,B,D]
        h = self.trans(x)  # [W,B,D]
        return self.fc(h[-1])  # [B,2]


# --------------------------- anomaly command -------------------------- #


def cmd_anomaly(args):
    X = np.vstack([np.loadtxt(f, delimiter=",", skiprows=1) for f in args.features])
    ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=args.nu)
    ocsvm.fit(X)
    joblib.dump(ocsvm, args.model)
    print(f"Saved anomaly model to {args.model}")


# --------------------------- train command --------------------------- #


def cmd_train(args):
    # 1) load & split
    ds = SequenceDataset(args.features, args.labels, args.window, args.model_type)
    X = ds.X.numpy()
    y = ds.y.numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 2) feature scaling
    if args.model_type == "ffnn":
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
    else:
        N, W, D = X_train.shape
        flat_train = X_train.reshape(N, -1)
        flat_val = X_val.reshape(X_val.shape[0], -1)
        scaler = StandardScaler().fit(flat_train)
        X_train = scaler.transform(flat_train).reshape(N, W, D)
        X_val = scaler.transform(flat_val).reshape(X_val.shape[0], W, D)
    # (Optional) save scaler: joblib.dump(scaler, args.model + ".scaler")

    # 3) XGBoost branch
    if args.use_xgb and args.model_type == "ffnn":
        neg, pos = np.bincount(y_train.astype(int))
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=neg / pos,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_val)
        y_prob = xgb.predict_proba(X_val)[:, 1]
        print("=== XGBoost Validation ===")
        print(confusion_matrix(y_val, y_pred))
        print(classification_report(y_val, y_pred, digits=4))
        print("ROC AUC:", roc_auc_score(y_val, y_prob))
        # Feature importances
        fi = xgb.feature_importances_
        names = [f"f{i}" for i in range(X_train.shape[1])]
        print("\nTop-10 feature importances:")
        for n, s in sorted(zip(names, fi), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {n}: {s:.4f}")
        joblib.dump(xgb, args.model)
        return

    # 4) DataLoader w/ WeightedRandomSampler
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    )
    counts = np.bincount(y_train.astype(int))
    class_w = 1.0 / counts
    sample_w = class_w[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w).float(),
        num_samples=len(sample_w),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # 5) model, optimizer, scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "ffnn":
        model = FFNet(X_train.shape[1]).to(device)
    elif args.model_type == "lstm":
        model = LSTMNet(X_train.shape[2]).to(device)
    else:
        model = TransformerNet(X_train.shape[2]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    no_improve = 0
    patience = 5

    # 6) train/validation loop with early stopping
    for ep in range(1, args.epochs + 1):
        # train
        model.train()
        train_loss_acc = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_acc += loss.item() * yb.size(0)
            n_train += yb.size(0)
        train_loss = train_loss_acc / n_train

        # validation
        model.eval()
        val_loss_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                val_loss_acc += loss.item() * yb.size(0)
                n_val += yb.size(0)
        val_loss = val_loss_acc / n_val

        print(
            f"Epoch[{ep}/{args.epochs}] "
            f"Train Loss:{train_loss:.4f}  Val Loss:{val_loss:.4f}"
        )

        # scheduler & early stop
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best Val Loss: {best_val_loss:.4f} → model saved to {args.model}")


# --------------------------- live command --------------------------- #


def cmd_live(args):
    ds = SequenceDataset(
        [args.features[0]], [args.labels[0]], args.window, args.model_type
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "ffnn":
        model = FFNet(ds.X.shape[1]).to(device)
    elif args.model_type == "lstm":
        model = LSTMNet(ds.X.shape[2]).to(device)
    else:
        model = TransformerNet(ds.X.shape[2]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    buf, prev_ts = [], 0.0

    def handle(pkt):
        nonlocal prev_ts
        if not pkt.haslayer(RadioTap) or not pkt.haslayer(Dot11):
            return
        if pkt[Dot11].type != 2:
            return
        fts, prev_ts = pkt_features(pkt[RadioTap], prev_ts)
        buf.append(fts)
        if len(buf) > args.window:
            buf.pop(0)
        if len(buf) == args.window:
            x = torch.tensor(buf, dtype=torch.float32).unsqueeze(0).to(device)
            if args.model_type == "ffnn":
                x = x.view(1, -1)
            if model(x).argmax(dim=1).item() == 1:
                print("[ALERT]", pkt.time)

    sniff(iface=args.iface, prn=handle, store=False)


# ----------------------------- main ------------------------------ #


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("-f", "--pcap", required=True)
    e.add_argument("-o", "--output", required=True)
    e.add_argument("-l", "--labels", required=True)
    e.add_argument("-w", "--window", type=int, default=11)
    e.add_argument("--label", type=int, default=1)
    e.add_argument(
        "--model-type", choices=("ffnn", "lstm", "transformer"), default="ffnn"
    )

    t = sub.add_parser("train")
    t.add_argument("--features", nargs="+", required=True)
    t.add_argument("--labels", nargs="+", required=True)
    t.add_argument("-m", "--model", default="wifi_dl.pth")
    t.add_argument("-w", "--window", type=int, default=11)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--use-xgb", action="store_true")
    t.add_argument(
        "--model-type", choices=("ffnn", "lstm", "transformer"), default="ffnn"
    )

    a = sub.add_parser("anomaly")
    a.add_argument("--features", nargs="+", required=True)
    a.add_argument("-m", "--model", default="anomaly_ocsvm.pkl")
    a.add_argument("--nu", type=float, default=0.05)

    l = sub.add_parser("live")
    l.add_argument("-i", "--iface", required=True)
    l.add_argument("-m", "--model", default="wifi_dl.pth")
    l.add_argument("-w", "--window", type=int, default=11)
    l.add_argument(
        "--model-type", choices=("ffnn", "lstm", "transformer"), default="ffnn"
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "anomaly":
        cmd_anomaly(args)
    elif args.cmd == "live":
        cmd_live(args)
    else:
        sys.exit("Unknown command")
