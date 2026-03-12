
import pandas as pd

raw_dataset = pd.read_csv("../data/iot_dataset_raw.csv")  

print(raw_dataset[['dns_rejected', 'ssl_resumed']].head(10))
print(raw_dataset[['dns_rejected', 'ssl_resumed']].value_counts().head(10))

