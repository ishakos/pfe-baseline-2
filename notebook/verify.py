import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ========================
# 1. Load dataset
# ========================

df = pd.read_csv("../data/iot_dataset_clean.csv")  
print("Original shape:", df.shape)
print(df.head())

# ========================
# 2. Drop unwanted columns
# ========================

print("Shape after dropping columns:", df.shape)
print("Remaining columns:", df.columns.tolist())

# ========================
# 3. Replace "-" with NaN
# ========================

print("Check for '-' remaining:", (df == "-").sum().sum())
print("NaN count per column after replacement:\n", df.isna().sum())

# ========================
# 4. Handle missing values
# ========================

print("Missing values after handling:")
print(df.isna().sum())

# ========================
# 5. Encode categorical columns
# ========================
# Using simple label encoding for demonstration

cat_cols = df.select_dtypes(include=["object"]).columns
cat_cols = cat_cols.drop("label", errors="ignore")

for col in cat_cols:
    print(f"Encoded {col}, unique values:", df[col].unique()[:10])

# ========================
# 6. Log transform large numeric features
# ========================

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
large_numeric_cols = []
for col in numeric_cols:
    skew = df[col].skew()
    if skew > 1.0:  
        large_numeric_cols.append(col)
        print(f"{col}: skew={skew:.2f}")

for col in large_numeric_cols:
    print(f"Applied log1p to {col}. Min/Max now:", df[col].min(), df[col].max())

# ========================
# 7. Remove duplicates
# ========================

after = df.shape[0]
print(f"Remaining duplicate rows: {after}")

