import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    brier_score_loss
)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import precision_recall_curve
from train import LSTMModel
from pipeline import create_sequences
from sklearn.model_selection import GroupShuffleSplit



# ===============================
# PREDICTION
# ===============================
def predict_lstm(model, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor).squeeze()
        probs = torch.sigmoid(logits)
        y_pred = (probs > 0.5).cpu().numpy()

    return y_pred, probs.cpu().numpy()


# ===============================
# CRITICAL CHECKS
# ===============================
def sanity_checks(y_test, y_pred):
    print("\n[CHECK] Class Distribution (y_test):")
    print(np.unique(y_test, return_counts=True))

    print("\n[CHECK] Prediction Distribution (y_pred):")
    print(np.unique(y_pred, return_counts=True))

    if len(np.unique(y_pred)) == 1:
        print("\n🚨 WARNING: MODEL PREDICTS ONLY ONE CLASS → INVALID MODEL")


# ===============================
# BASELINE
# ===============================
def baseline_sanity(model, X_test, y_test, X_train=None, y_train=None):
    report = {}

    dummy = DummyClassifier(strategy="most_frequent")

    if X_train is not None:
        dummy.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    else:
        dummy.fit(X_test.reshape(X_test.shape[0], -1), y_test)

    dummy_pred = dummy.predict(X_test.reshape(X_test.shape[0], -1))
    report["dummy_f1"] = f1_score(y_test, dummy_pred)

    preds, _ = predict_lstm(model, X_test)
    report["model_f1"] = f1_score(y_test, preds)

    if X_train is not None:
        train_preds, _ = predict_lstm(model, X_train)
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, preds)

        report["train_acc"] = train_acc
        report["test_acc"] = test_acc
        report["train_test_gap"] = train_acc - test_acc

    return report


# ===============================
# ADVANCED METRICS
# ===============================
def advanced_metrics(model, X_test, y_test):
    preds, probs = predict_lstm(model, X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
        "brier_score": brier_score_loss(y_test, probs),

        "confusion_matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        }
    }


# ===============================
# ROBUSTNESS
# ===============================
def robustness_tests(model, X_test, y_test):
    results = {}

    preds, _ = predict_lstm(model, X_test)
    base_f1 = f1_score(y_test, preds)

    # Noise
    noise = np.random.normal(0, 0.05, X_test.shape)
    X_noise = X_test + noise
    preds_noise, _ = predict_lstm(model, X_noise)
    results["noise_f1"] = f1_score(y_test, preds_noise)

    # Outliers
    X_out = X_test.copy()
    idx = np.random.choice(len(X_out), int(0.05 * len(X_out)), replace=False)
    X_out[idx] *= 10
    preds_out, _ = predict_lstm(model, X_out)
    results["outlier_f1"] = f1_score(y_test, preds_out)

    # Missing
    X_nan = X_test.copy()
    idx = np.random.choice(len(X_nan), int(0.05 * len(X_nan)), replace=False)
    X_nan[idx] = 0
    preds_nan, _ = predict_lstm(model, X_nan)
    results["nan_f1"] = f1_score(y_test, preds_nan)

    return results


# ===============================
# DATA LEAKAGE CHECK
# ===============================
def leakage_check(X_train, X_test):
    train_flat = X_train.reshape(X_train.shape[0], -1)
    test_flat = X_test.reshape(X_test.shape[0], -1)

    df_train = pd.DataFrame(train_flat)
    df_test = pd.DataFrame(test_flat)

    merged = df_train.merge(df_test)

    print(f"\n[LEAKAGE CHECK] Overlapping samples: {len(merged)}")

    if len(merged) > 0:
        print("🚨 WARNING: DATA LEAKAGE DETECTED")


# ===============================
# MAIN
# ===============================
def diagnose_model(model_path, X_test, y_test, X_train=None, y_train=None):
    model = joblib.load(model_path)

    sanity_checks(y_test, predict_lstm(model, X_test)[0])

    if X_train is not None:
        leakage_check(X_train, X_test)

    report = {}

    report["baseline"] = baseline_sanity(model, X_test, y_test, X_train, y_train)
    report["metrics"] = advanced_metrics(model, X_test, y_test)
    report["robustness"] = robustness_tests(model, X_test, y_test)

    print("\n🚨 LSTM MODEL HEALTH REPORT 🚨")
    print("=" * 50)
    print(json.dumps(report, indent=4))

    return report


# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.data)

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

    y = df["label"].values

    X = X.values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train_raw = X[train_idx]
    X_test_raw  = X[test_idx]

    y_train_raw = y[train_idx]
    y_test_raw  = y[test_idx]

    groups_train = groups[train_idx]
    groups_test  = groups[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    X_train, y_train = create_sequences(X_train_scaled, y_train_raw, groups_train)
    X_test, y_test   = create_sequences(X_test_scaled, y_test_raw, groups_test)

    # DUPLICATE CHECK
    print(f"\n[DATA CHECK] Duplicates: {df.duplicated().sum()}")

    diagnose_model(args.model, X_test, y_test, X_train, y_train)