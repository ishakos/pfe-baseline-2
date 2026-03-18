import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_dataset(path):

    df = pd.read_csv(path)
    df = df.sort_values(by=["src_ip"]).reset_index(drop=True)
    groups = df["src_ip"].values

    X = df.drop(columns=[
        "src_ip",
        'src_bytes', 
        'dst_bytes',
        'src_ip_bytes', 
        'dst_ip_bytes', 
        'http_version',
        'http_method',
        'ssl_resumed',
        "label",
        "type",
    ], errors="ignore")

    X = X.values
    
    y = df["label"].values

    return X, y, groups

def create_sequences(X, y, groups, seq_len=10):
    sequences = []
    labels = []

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]

        X_g = X[idx]
        y_g = y[idx]

        for i in range(len(X_g) - seq_len):
            sequences.append(X_g[i:i+seq_len])
            labels.append(y_g[i+seq_len])

    return np.array(sequences), np.array(labels)