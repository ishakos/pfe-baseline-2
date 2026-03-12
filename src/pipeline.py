import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_dataset(path):

    df = pd.read_csv(path)

    X = df.drop(columns=["label", "type"])
    y = df["label"]

    return X, y


def scale_features(X):

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = np.nan_to_num(
        X_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return X_scaled, scaler


def create_sequences(X, y, window_size=10):

    X_seq = []
    y_seq = []

    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])

    return np.array(X_seq), np.array(y_seq)