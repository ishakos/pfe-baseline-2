import torch
from pipeline import load_dataset, create_sequences
from train import train_model
from evaluate import evaluate_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = "../../Data/iot_dataset_clean.csv"

# 1. Load (WITH GROUPS)
X, y, groups = load_dataset(DATA_PATH)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, test_idx = next(gss.split(X, y, groups))

X_train_raw = X[train_idx]
X_test_raw  = X[test_idx]

y_train_raw = y[train_idx]
y_test_raw  = y[test_idx]

groups_train = groups[train_idx]
groups_test  = groups[test_idx]

# 3. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

# 4. Create GROUPED sequences
X_train, y_train = create_sequences(X_train_scaled, y_train_raw, groups_train)
X_test, y_test   = create_sequences(X_test_scaled, y_test_raw, groups_test)

# 5. Train
model = train_model(X_train, y_train)

# 6. Evaluate
evaluate_model(model, X_test, y_test)