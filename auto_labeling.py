import joblib, numpy as np

ocsvm = joblib.load("anomaly_ocsvm.pkl")
X_suspicious = np.loadtxt("train_pcap/feat_susp.csv", delimiter=",")
scores = ocsvm.decision_function(X_suspicious)

low_thresh = np.percentile(scores, 1)
high_thresh = np.percentile(scores, 80)

y_auto_label = np.full(scores.shape, -1)
y_auto_label[scores <= low_thresh] = 1
y_auto_label[scores >= high_thresh] = 0

selected_X = X_suspicious[y_auto_label != -1]
selected_y = y_auto_label[y_auto_label != -1]

np.savetxt("auto_labeled_X.csv", selected_X, delimiter=",")
np.savetxt("auto_labeled_y.dat", selected_y, fmt="%d")

print("정상 개수:", np.sum(selected_y == 0))
print("이상 개수:", np.sum(selected_y == 1))
