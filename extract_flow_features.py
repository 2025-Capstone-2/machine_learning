#!/usr/bin/env python3
# extract_flow_features.py

import argparse
import numpy as np
import csv
from collections import defaultdict
from scapy.all import PcapReader, Dot11, RadioTap, IP, TCP, UDP


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract flow-level features from one or more PCAPs"
    )
    p.add_argument(
        "--pcaps",
        nargs="+",
        required=True,
        help="PCAP files to process (order matches --labels)",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Integer label for each PCAP (e.g. 1 0 0)",
    )
    p.add_argument(
        "-o", "--output", required=True, help="CSV file in which to write flow features"
    )
    p.add_argument(
        "-l",
        "--labelfile",
        required=True,
        help="File in which to write one label per flow",
    )
    return p.parse_args()


def extract_flow_features(pcap_files, label_values):
    # flows: key=(5-tuple, direction flag ignored here), value=list of (ts,len,dir)
    all_feats = []
    all_labels = []

    for pcap, lbl in zip(pcap_files, label_values):
        flows = defaultdict(list)
        for pkt in PcapReader(pcap):
            if not pkt.haslayer(RadioTap) or not pkt.haslayer(Dot11):
                continue
            dot11 = pkt[Dot11]
            if dot11.type != 2:  # Data frame
                continue

            # IP 대신 MAC 주소 추출
            src_mac = dot11.addr2  # transmitter
            dst_mac = dot11.addr1  # receiver
            # toDS 비트(프레임이 AP로 가는지)
            to_ds = 1 if (dot11.FCfield & 0x01) else 0

            # 흐름 키: (송신 MAC, 수신 MAC, toDS)
            key = (src_mac, dst_mac, to_ds)

            # 시간/크기 float으로 변환해서 저장
            ts = float(pkt.time)
            length = float(len(pkt))
            flows[key].append((ts, length, to_ds))

        # now compute per-flow stats
        for key, pkts in flows.items():
            times = np.array([t for t, _, _ in pkts])
            sizes = np.array([s for _, s, _ in pkts])
            dirs = np.array([d for _, _, d in pkts])
            if len(times) < 2:
                continue
            inter = np.diff(times)

            duration = float(times[-1] - times[0])
            total_bytes = float(sizes.sum())
            up_ratio = float(dirs.mean())
            p10, p50, p90 = np.percentile(sizes, [10, 50, 90]).tolist()
            jitter_mean = float(inter.mean())
            jitter_var = float(inter.var())
            std = float(inter.std()) if inter.std() > 0 else 1.0
            skew = float(((inter - inter.mean()) ** 3).mean() / (std**3))
            kurt = float(((inter - inter.mean()) ** 4).mean() / (std**4) - 3)

            feat = [
                duration,
                total_bytes,
                up_ratio,
                p10,
                p50,
                p90,
                jitter_mean,
                jitter_var,
                skew,
                kurt,
            ]
            all_feats.append(feat)
            all_labels.append(lbl)

    return all_feats, all_labels


def main():
    args = parse_args()
    # parse labels as ints
    labels = [int(x) for x in args.labels]
    feats, labs = extract_flow_features(args.pcaps, labels)

    # write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        # header
        writer.writerow(
            [
                "duration",
                "total_bytes",
                "up_ratio",
                "p10",
                "p50",
                "p90",
                "jitter_mean",
                "jitter_var",
                "skew",
                "kurt",
            ]
        )
        writer.writerows(feats)

    # write labels
    with open(args.labelfile, "w") as f:
        for l in labs:
            f.write(f"{l}\n")

    print(f"Wrote {len(feats)} flows → {args.output}, {args.labelfile}")


if __name__ == "__main__":
    main()
