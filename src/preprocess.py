# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# 1. Load dataset
# =========================

raw_dataset = pd.read_csv("../data/iot_dataset_raw.csv")

# =========================
# 2. Drop identifier columns
# =========================

cols_to_drop = ["src_ip", "dst_ip", "src_port", "dst_port", "conn_state", "service", "dns_query", "dns_AA", "dns_RD", "dns_RA", "dns_rcode", "ssl_subject", "ssl_issuer", "ssl_established", "http_uri", "http_user_agent", "http_orig_mime_types", "http_resp_mime_types", "http_status_code", "weird_addl", "weird_name", "weird_notice"]

col_to_keep = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes", "missed_bytes", "proto", "dns_qclass", "dns_qtype", "dns_rejected", "ssl_version", "ssl_cipher", "ssl_resumed", "http_method", "http_version", "http_trans_depth", "http_request_body_len", "http_response_body_len", "label", "type"]

raw_dataset = raw_dataset.drop(columns=cols_to_drop, errors="ignore")

# =========================
# 3. Replace "-" with NaN
# =========================

raw_dataset = raw_dataset.replace("-", np.nan)

# =========================
# 4. Handle missing values
# =========================

# Numerical columns
num_cols = raw_dataset.select_dtypes(include=["int64", "float64"]).columns
raw_dataset[num_cols] = raw_dataset[num_cols].fillna(0)

# Categorical columns
cat_cols = raw_dataset.select_dtypes(include=["object"]).columns
raw_dataset[cat_cols] = raw_dataset[cat_cols].fillna("Unknown")

# =========================
# 5. Convert T/F columns to binary (Encoding boolean columns)
# =========================

bool_cols = [
    "dns_rejected",
    "ssl_resumed",
    "ssl_established",
]

for col in bool_cols:
    if col in raw_dataset.columns:
        raw_dataset[col] = raw_dataset[col].map({"T": 1, "F": 0})

# =========================
# 6. Encode categorical columns
# =========================

cat_cols = raw_dataset.select_dtypes(include=["object"]).columns
cat_cols = cat_cols.drop("label", errors="ignore")

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    raw_dataset[col] = le.fit_transform(raw_dataset[col])
    encoders[col] = le

# =========================
# 7. Log transform large numeric features 
# (Normalization but for extreme large numbers)
# =========================

log_cols = [
    "src_bytes",
    "dst_bytes",
    "src_ip_bytes",
    "dst_ip_bytes"
]

for col in log_cols:
    if col in raw_dataset.columns:
        raw_dataset[col] = np.log1p(raw_dataset[col])

# =========================
# 8. Remove duplicates
# =========================

raw_dataset = raw_dataset.drop_duplicates().reset_index(drop=True)

# =========================
# 9. Save cleaned dataset
# =========================

raw_dataset.to_csv("../data/iot_dataset_clean2.csv", index=False)

print("Clean dataset saved successfully.")