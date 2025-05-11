# evaluate_wifi_model.py

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# stream_video_wifi_detector.py 에 정의된 것들
from stream_video_wifi_detector import SequenceDataset, FFNet, LSTMNet, TransformerNet


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Wi-Fi streaming video model"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="feature CSV files (windowed output)",
    )
    parser.add_argument(
        "--labels", nargs="+", required=True, help="corresponding label files"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="path to saved model (.pth)"
    )
    parser.add_argument(
        "--window", type=int, default=11, help="window size used during extract/train"
    )
    parser.add_argument(
        "--model-type",
        choices=("ffnn", "lstm", "transformer"),
        default="ffnn",
        help="which architecture was trained",
    )
    parser.add_argument(
        "--batch", type=int, default=128, help="batch size for evaluation"
    )
    args = parser.parse_args()

    # 1) Dataset + DataLoader
    ds = SequenceDataset(args.features, args.labels, args.window, args.model_type)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    # 2) Model load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "ffnn":
        model = FFNet(ds.X.shape[1]).to(device)
    elif args.model_type == "lstm":
        pkt_dim = ds.X.shape[2]
        model = LSTMNet(pkt_dim).to(device)
    else:  # transformer
        pkt_dim = ds.X.shape[2]
        model = TransformerNet(pkt_dim).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # 3) Inference + metrics
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)

            # FFNN gets flat input, LSTM/Trans gets [B,W,feat]
            if args.model_type == "ffnn":
                logits = model(Xb)
            else:
                logits = model(Xb)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = np.argmax(logits.cpu().numpy(), axis=1)

            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    # 4) Report
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_true, y_prob))


if __name__ == "__main__":
    main()
