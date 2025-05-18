import pandas as pd
import matplotlib.pyplot as plt

# 1) Load features only (no labels)
features = pd.read_csv("suspicious_features.csv", header=None)

# 2) Assign full column names
orig_feat_names = [
    "mean_inter",
    "var_inter",
    "mean_len",
    "var_len",
    "mean_sig",
    "var_sig",
    "mean_rate",
    "var_rate",
    "mean_retry",
    "var_retry",
]
stats_names = [
    "duration",
    "byte_count",
    "p25",
    "p50",
    "p75",
    "p90",
    "ul_dl_ratio",
    "jitter_mean",
    "jitter_var",
    "size_mean",
    "size_var",
    "size_skew",
    "ctrl_ratio",
]
all_column_names = orig_feat_names + stats_names
features.columns = all_column_names

# 3) Select only relevant columns to visualize
target_cols = [
    "duration",
    "byte_count",
    "p75",
    "ul_dl_ratio",
    "jitter_var",
    "ctrl_ratio",
]
df_sel = features[target_cols]

# 4) Plot histogram for each parameter
for col in target_cols:
    plt.figure()
    df_sel[col].plot(kind="hist", bins=50, alpha=0.7)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

subset.to_csv("subset_features.csv", index=False)
