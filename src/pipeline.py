import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_dataset(path):

    df = pd.read_csv(path)

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
    
    y = df["label"]

    return X, y

def create_sequences(X, y, window_size=10):

    X_seq = []
    y_seq = []

    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])

    return np.array(X_seq), np.array(y_seq)