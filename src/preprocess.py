# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# =========================
# 1. Load dataset
# =========================

raw_dataset = pd.read_csv("../data/iot_dataset_raw.csv")
print("Original shape:", raw_dataset.shape)
print(raw_dataset.head())

# =========================
# 2. Drop identifier columns
# =========================

cols_to_drop = ["src_ip", "dst_ip", "src_port", "dst_port", "conn_state", "service", "dns_query", "dns_AA", "dns_RD", "dns_RA", "dns_rcode", "ssl_subject", "ssl_issuer", "ssl_established", "http_uri", "http_user_agent", "http_orig_mime_types", "http_resp_mime_types", "http_status_code", "weird_addl", "weird_name", "weird_notice"]

col_to_keep = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes", "missed_bytes", "proto", "dns_qclass", "dns_qtype", "dns_rejected", "ssl_version", "ssl_cipher", "ssl_resumed", "http_method", "http_version", "http_trans_depth", "http_request_body_len", "http_response_body_len", "label", "type"]

raw_dataset = raw_dataset.drop(columns=cols_to_drop, errors="ignore")

print("Shape after dropping columns:", raw_dataset.shape)
print("Remaining columns:", raw_dataset.columns.tolist())

# =========================
# 3. Replace "-" with NaN
# =========================

raw_dataset = raw_dataset.replace("-", np.nan)

print("Check for '-' remaining:", (raw_dataset == "-").sum().sum())
print("NaN count per column after replacement:\n", raw_dataset.isna().sum())

# =========================
# 4. Handle missing values
# =========================

# Numerical columns
num_cols = raw_dataset.select_dtypes(include=["int64", "float64"]).columns
raw_dataset[num_cols] = raw_dataset[num_cols].fillna(0)

# Categorical columns
cat_cols = raw_dataset.select_dtypes(include=["object"]).columns
raw_dataset[cat_cols] = raw_dataset[cat_cols].fillna("Unknown")

print("Missing values after handling:")
print(raw_dataset.isna().sum())

# =========================
# 5. Encode categorical columns
# =========================

cat_cols = raw_dataset.select_dtypes(include=["object", "string"]).columns
cat_cols = cat_cols.drop("label", errors="ignore")

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    raw_dataset[col] = le.fit_transform(raw_dataset[col])
    encoders[col] = le
    print(f"Encoded {col}, unique values:", raw_dataset[col].unique()[:10])

# =========================
# 6. Log transform large numeric features 
# (Normalization but for extreme large numbers)
# =========================

numeric_cols = raw_dataset.select_dtypes(include=["int64", "float64"]).columns
large_numeric_cols = []
for col in numeric_cols:
    skew = raw_dataset[col].skew()
    if skew > 1.0:  
        large_numeric_cols.append(col)
        print(f"{col}: skew={skew:.2f}")
print("Will log transform:", large_numeric_cols)

for col in large_numeric_cols:
    raw_dataset[col] = np.log1p(raw_dataset[col])
    print(f"Applied log1p to {col}. Min/Max now:", raw_dataset[col].min(), raw_dataset[col].max())

# =========================
# 7. Remove duplicates
# =========================

before = raw_dataset.shape[0]
raw_dataset = raw_dataset.drop_duplicates().reset_index(drop=True)
after = raw_dataset.shape[0]
print(f"Removed {before - after} duplicate rows. Remaining rows: {after}")

# =========================
# 8. Save cleaned dataset
# =========================

raw_dataset.to_csv("../data/iot_dataset_clean.csv", index=False)

print("Clean dataset saved successfully.")

# ========================
# 9. Validation: reload and check
# ========================
df_check = pd.read_csv("../data/iot_dataset_clean.csv")
print("Reloaded shape:", df_check.shape)
print(df_check.head())
print("Check for duplicates in saved file:", df_check.duplicated().sum())
print("Check for missing values in saved file:\n", df_check.isna().sum())