
import pandas as pd

df = pd.read_csv("../../Data/iot_dataset_raw.csv")  

print("All features:", df.columns.tolist())
print("\nTotal columns:", len(df.columns))
print("\nFirst few rows shape:", df.shape)

# Quick check for timestamp
if 'ts' in df.columns:
    print("\n✅ Timestamp 'ts' exists")
    print("Sample:", df['ts'].head())

