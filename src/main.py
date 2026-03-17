from sklearn.model_selection import train_test_split
import torch
from pipeline import load_dataset, create_sequences
from train import train_model
from evaluate import evaluate_model


DATA_PATH = "../../Data/iot_dataset_clean.csv"


# 1. Load
X, y = load_dataset(DATA_PATH)

# 1. Split FIRST
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X,
    y.values,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 2. Scale (fit only on train)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 3. Create sequences separately
X_train, y_train = create_sequences(X_train_scaled, y_train_raw)
X_test, y_test = create_sequences(X_test_scaled, y_test_raw)

# 4. Train
model = train_model(X_train, y_train)

# we'll do later
X_small = X_train[:500]
y_small = y_train[:500]

# model = train_model(X_small, y_small)

evaluate_model(model, X_test, y_test)