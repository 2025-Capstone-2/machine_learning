# evaluate_anomaly_suspicious.py
import joblib, numpy as np

ocsvm = joblib.load("anomaly_ocsvm.pkl")
X_suspicious = np.loadtxt("train_pcap/feat_susp.csv", delimiter=",")

pred = ocsvm.predict(X_suspicious)
anomaly_indices = np.where(pred == -1)[0]  # 이상치(-1) 추출

print("Detected anomaly indices:", anomaly_indices)
