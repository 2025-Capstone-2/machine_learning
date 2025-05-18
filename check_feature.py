import pandas as pd
import matplotlib.pyplot as plt

# 1. CSV 로딩
# features와 labels 파일을 각각 읽음
features = pd.read_csv("train_pcap/suspicious_features.csv")
labels = pd.read_csv("train_pcap/suspicious_labels.csv", header=None, names=["label"])

# 두 파일을 옆으로 합침 (행 순서가 반드시 일치해야 함)
df = pd.concat([features, labels], axis=1)

# 2. 주요 특징 컬럼(예시, 실제 컬럼명에 맞게 수정)
feature_cols = [
    "stream_key",
    "total_inbound",
    "total_outbound",
    "mean_interval",
    "std_interval",
    "pkt_count",
    "label",
]

# 3. 라벨별로 분리
df_cam = df[df["label"] == 1]
df_normal = df[df["label"] == 0]

# 4. 주요 feature별 기술통계(평균, std, min, max 등)
print("==== Suspicious stream features ====")
print(df_cam[feature_cols].describe())
print("\n==== Normal stream features ====")
print(df_normal[feature_cols].describe())

# print(df[df["label"] == 1])
print("\n\n")
print(df[df["label"] == 1][feature_cols].describe())

for col in feature_cols:
    plt.figure(figsize=(6, 4))
    plt.hist(df[df["label"] == 1][col], bins=20, alpha=0.5, label="label=1")
    plt.hist(df[df["label"] == 0][col], bins=20, alpha=0.5, label="label=0")
    plt.title(f"{col} (label Distribution)")
    plt.legend()
    plt.show()

# 5. 시각화: feature별로 라벨별 히스토그램/박스플롯
# for col in feature_cols:
#     plt.figure(figsize=(6, 4))
#     plt.hist(df_cam[col], bins=20, alpha=0.5, label="Suspicious stream(label=1)")
#     plt.hist(df_normal[col], bins=20, alpha=0.5, label="Normal stream(label=0)")
#     plt.title(f"{col} Distribution")
#     plt.legend()
#     plt.show()
