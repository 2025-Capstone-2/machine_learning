# stream_video_wifi_detector.py
"""
Encrypted Wi-Fi (802.11) Streaming Video Detector
Supports feature extraction, deep-learning training, and live detection.
Modes:
  extract - extract features + labels from monitor-mode .pcap
  train   - train deep learning model on extracted data
  live    - sniff live 802.11 packets and classify suspicious streams
"""
import argparse, csv, joblib, sys
import numpy as np
from scapy.all import PcapReader, sniff, RadioTap, Dot11
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

# ----------------------- Feature Extraction ----------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    # extract
    e = sub.add_parser('extract')
    e.add_argument('-f','--pcap', required=True)
    e.add_argument('-o','--output', required=True)
    e.add_argument('-l','--labels', required=True)
    e.add_argument('-w','--window', type=int, default=11)
    e.add_argument('--label', type=int, default=1)
    # train
    t = sub.add_parser('train')
    t.add_argument('--features', nargs='+', required=True)
    t.add_argument('--labels', nargs='+', required=True)
    t.add_argument('-m','--model', default='wifi_dl.pth')
    t.add_argument('--epochs', type=int, default=10)
    t.add_argument('--batch', type=int, default=64)
    # live
    l = sub.add_parser('live')
    l.add_argument('-i','--iface', required=True)
    l.add_argument('-m','--model', default='wifi_dl.pth')
    l.add_argument('-w','--window', type=int, default=11)
    return p.parse_args()

# feature routines

def pkt_features(pkt, prev_ts):
    ts = pkt.time
    inter = ts - prev_ts if prev_ts else 0.0
    length = len(pkt)
    subtype = pkt.subtype if hasattr(pkt, 'subtype') else 0
    seqnum = (pkt.SC >> 4) & 0xFFF if hasattr(pkt, 'SC') else 0
    sig = pkt.dBm_AntSignal if hasattr(pkt, 'dBm_AntSignal') and pkt.dBm_AntSignal is not None else 0.0
    rate = pkt.Rate if hasattr(pkt, 'Rate') and pkt.Rate is not None else 0.0
    retry = 1 if hasattr(pkt, 'FCfield') and pkt.FCfield & 0x08 else 0
    return [inter, length, subtype, seqnum, sig, rate, retry], ts


def window_stats(samples):
    inter = [float(s[0]) for s in samples]
    length = [int(s[1]) for s in samples]
    sig = [float(s[4]) for s in samples]
    rate = [float(s[5]) for s in samples]
    retry = [int(s[6]) for s in samples]
    stats = []
    for arr in (inter, length, sig, rate):
        stats.append(float(np.mean(arr)))
        stats.append(float(np.var(arr)))
    stats.append(sum(retry))
    return stats

# -------------------------- Dataset ------------------------------- #
class WiFiDataset(Dataset):
    def __init__(self, feature_files, label_files):
        X_list = [np.loadtxt(f, delimiter=',') for f in feature_files]
        y_list = [np.loadtxt(l) for l in label_files]
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --------------------------- Model -------------------------------- #
class WiFiNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

# --------------------------- Commands ----------------------------- #

def cmd_extract(args):
    reader = PcapReader(args.pcap)
    feats_f = open(args.output,'w',newline='')
    writer = csv.writer(feats_f)
    label_f = open(args.labels,'w')
    samples, prev_ts = [], 0.0
    for pkt in reader:
        if not pkt.haslayer(RadioTap) or not pkt.haslayer(Dot11): continue
        dot11 = pkt[Dot11]
        if dot11.type!=2: continue
        fts, prev_ts = pkt_features(pkt[RadioTap], prev_ts)
        samples.append(fts)
    reader.close()

    n=args.window
    for i in range(len(samples)-n+1):
        stat = window_stats(samples[i:i+n])
        writer.writerow(stat)
        label_f.write(f"{args.label}\n")
    feats_f.close(); label_f.close()
    print(f"Extracted {len(samples)-n+1} windows")


def cmd_train(args):
    ds = WiFiDataset(args.features, args.labels)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    model = WiFiNet(ds.X.shape[1]).to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = next(model.parameters()).device
    for ep in range(args.epochs):
        model.train(); total=0; loss_acc=0
        for X,y in loader:
            X,y=X.to(device),y.to(device)
            optimizer.zero_grad()
            logits=model(X)
            loss=criterion(logits,y)
            loss.backward(); optimizer.step()
            total+=y.size(0); loss_acc+=loss.item()*y.size(0)
        print(f"Epoch [{ep+1}/{args.epochs}] Loss: {loss_acc/total:.4f}")
    torch.save(model.state_dict(), args.model)
    print(f"Saved DL model to {args.model}")


def cmd_live(args):
    # load DL model
    ds = WiFiDataset([args.features[0]],[args.labels[0]])  # dummy to get dim
    model = WiFiNet(ds.X.shape[1])
    model.load_state_dict(torch.load(args.model))
    model.eval(); device=next(model.parameters()).device
    buf, prev_ts = [],0.0
    def handle(pkt):
        nonlocal prev_ts
        if not pkt.haslayer(RadioTap) or not pkt.haslayer(Dot11): return
        dot11=pkt[Dot11];
        if dot11.type!=2: return
        fts, prev_ts = pkt_features(pkt[RadioTap], prev_ts)
        buf.append(fts)
        if len(buf)>args.window: buf.pop(0)
        if len(buf)==args.window:
            stat=window_stats(buf)
            x=torch.tensor(stat,dtype=torch.float32).unsqueeze(0)
            logits=model(x);
            pred=logits.argmax(dim=1).item()
            if pred==1: print('[ALERT] Suspicious at', pkt.time)
    sniff(iface=args.iface, prn=handle, store=False)

# ---------------------------- Main ------------------------------- #
if __name__=='__main__':
    args=parse_args()
    if args.cmd=='extract': cmd_extract(args)
    elif args.cmd=='train':   cmd_train(args)
    elif args.cmd=='live':    cmd_live(args)
    else: print('Unknown command')
