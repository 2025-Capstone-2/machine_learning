from sklearn.metrics import classification_report, confusion_matrix
import joblib, numpy as np

# 1) 피처 / 라벨 로드 (test_features.csv 는 정상+이상 모두 포함)
X_test = np.loadtxt("test_features.csv", delimiter=",")
y_test = np.loadtxt("test_labels.dat")  # 0=normal, 1=anomaly

# 2) 모델 로드
ocsvm = joblib.load("anomaly_ocsvm.pkl")

# 3) 예측: inliers=+1, outliers=-1
pred = ocsvm.predict(X_test)
# 편리하게 0/1 로 변환 (1=normal→0, -1=anomaly→1)
y_pred = np.where(pred == 1, 0, 1)

# 4) 지표 출력
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
