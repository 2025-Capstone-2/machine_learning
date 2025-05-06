import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from stream_video_wifi_detector import WiFiDataset, WiFiNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Wi-Fi streaming detector")
    parser.add_argument('--features', nargs='+', required=True,
                        help='List of feature CSV files')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='List of label files (dat)')
    parser.add_argument('-m', '--model', default='wifi_dl.pth',
                        help='Path to trained model file')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size for evaluation')
    return parser.parse_args()


def main():
    args = parse_args()
    # Load dataset
    ds = WiFiDataset(args.features, args.labels)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WiFiNet(ds.X.shape[1]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Evaluate
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(y.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    # Metrics
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_true, y_prob))


if __name__ == '__main__':
    main()
