import argparse
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    brier_score_loss, precision_recall_curve
)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
import json

from train import LSTMModel, train_model
from pipeline import create_sequences


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

    return probs.cpu().numpy()


# ===============================
# THRESHOLD OPTIMIZATION
# ===============================
def find_best_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)

    return thresholds[best_idx], f1_scores[best_idx]


# ===============================
# SANITY CHECKS
# ===============================
def sanity_checks(y_test, preds):
    print("\n[CHECK] Class Distribution:")
    print(np.unique(y_test, return_counts=True))

    print("\n[CHECK] Prediction Distribution:")
    print(np.unique(preds, return_counts=True))

    if len(np.unique(preds)) == 1:
        print("\n🚨 WARNING: MODEL PREDICTS ONLY ONE CLASS")


# ===============================
# BASELINES
# ===============================
def baseline_sanity(model, X_test, y_test, X_train, y_train, probs, threshold):
    report = {}

    # Dummy baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    dummy_pred = dummy.predict(X_test.reshape(X_test.shape[0], -1))
    report["dummy_f1"] = f1_score(y_test, dummy_pred)

    # Logistic baseline
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_flat, y_train)
    logreg_pred = logreg.predict(X_test_flat)

    report["logreg_f1"] = f1_score(y_test, logreg_pred)

    # LSTM
    preds = (probs > threshold)
    report["model_f1"] = f1_score(y_test, preds)

    train_probs = predict_lstm(model, X_train)
    train_preds = (train_probs > threshold)

    report["train_acc"] = accuracy_score(y_train, train_preds)
    report["test_acc"] = accuracy_score(y_test, preds)
    report["train_test_gap"] = report["train_acc"] - report["test_acc"]

    return report


# ===============================
# METRICS
# ===============================
def advanced_metrics(y_test, probs, threshold):
    preds = (probs > threshold)

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
def robustness_tests(model, X_test, y_test, threshold):
    results = {}

    def eval_variant(X_variant):
        probs = predict_lstm(model, X_variant)
        preds = (probs > threshold)
        return f1_score(y_test, preds)

    # Noise
    noise = np.random.normal(0, 0.05, X_test.shape)
    results["noise_f1"] = eval_variant(X_test + noise)

    # Outliers
    X_out = X_test.copy()
    idx = np.random.choice(len(X_out), int(0.05 * len(X_out)), replace=False)
    X_out[idx] *= 10
    results["outlier_f1"] = eval_variant(X_out)

    # Missing
    X_nan = X_test.copy()
    idx = np.random.choice(len(X_nan), int(0.05 * len(X_nan)), replace=False)
    X_nan[idx] = 0
    results["nan_f1"] = eval_variant(X_nan)

    return results


# ===============================
# LEAKAGE
# ===============================
def leakage_check(X_train, X_test):
    train_flat = X_train.reshape(X_train.shape[0], -1)
    test_flat = X_test.reshape(X_test.shape[0], -1)

    df_train = pd.DataFrame(train_flat)
    df_test = pd.DataFrame(test_flat)

    overlap = len(df_train.merge(df_test))
    print(f"\n[LEAKAGE CHECK] Overlap: {overlap}")

    if overlap > 0:
        print("🚨 WARNING: DATA LEAKAGE")

# ===============================
# OTHERS
# ===============================

def class_wise_performance(y_test, preds):
    print("\n[CLASS-WISE PERFORMANCE]")
    print(confusion_matrix(y_test, preds))

    from sklearn.metrics import classification_report
    print("\n[DETAILED CLASSIFICATION REPORT]")
    print(classification_report(y_test, preds))

def bootstrap_f1(y_true, preds, n=1000):
    scores = []
    for _ in range(n):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        scores.append(f1_score(y_true[idx], preds[idx]))
    return np.mean(scores), np.std(scores)

from sklearn.model_selection import GroupKFold

def group_kfold_cv(model_path, df, n_splits=5):  
    """GroupKFold CV on RAW data. Fixes NaN AUCs."""
    model = joblib.load(model_path)
    groups = df["src_ip"].values
    gkf = GroupKFold(n_splits=n_splits)
    
    f1_scores, auc_scores = [], []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        # RAW fold data
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
        
        # EXACT same feature selection as main pipeline
        drop_cols = [
            "src_ip", "src_bytes", "dst_bytes", "src_ip_bytes", "dst_ip_bytes",
            "http_version", "http_method", "ssl_resumed", "label", "type"
        ]
        X_train_raw = df_train.drop(columns=drop_cols, errors="ignore").values
        X_test_raw = df_test.drop(columns=drop_cols, errors="ignore").values
        y_train_raw, y_test_raw = df_train["label"].values, df_test["label"].values
        g_train, g_test = groups[train_idx], groups[test_idx]
        
        # Scale + sequence
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, g_train)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw, g_test)
        
        # Predict
        probs = predict_lstm(model, X_test_seq)
        best_thresh, _ = find_best_threshold(y_test_seq, probs)
        preds = (probs > best_thresh)
        
        # F1 always works
        f1_scores.append(f1_score(y_test_seq, preds))
        
        # AUC only if both classes present (fixes NaN)
        if len(np.unique(y_test_seq)) > 1:
            auc_scores.append(roc_auc_score(y_test_seq, probs))
            auc_val = auc_scores[-1]
        else:
            auc_scores.append(np.nan)
            auc_val = "no_pos"
        
        print(f"Fold {fold+1}: F1={f1_scores[-1]:.3f}, AUC={auc_val}")
    
    # Clean NaNs for mean/std
    valid_auc = [x for x in auc_scores if not np.isnan(x)]
    auc_mean = np.mean(valid_auc) if valid_auc else np.nan
    auc_std = np.std(valid_auc) if len(valid_auc) > 1 else np.nan
    
    return {
        "f1": [np.mean(f1_scores), np.std(f1_scores)],
        "auc": [auc_mean, auc_std],
        "f1_per_fold": [f"{f:.3f}" for f in f1_scores],
        "note": f"{len(auc_scores) - len(valid_auc)} folds had no positive class"
    }


# ===============================
# MAIN DIAGNOSIS
# ===============================
def diagnose_model(model_path, df, X_test, y_test, X_train, y_train):
    model = joblib.load(model_path)

    probs = predict_lstm(model, X_test)

    # 🔥 THRESHOLD FIX
    best_threshold, best_f1 = find_best_threshold(y_test, probs)

    print("\n[THRESHOLD OPTIMIZATION]")
    print("Best threshold:", best_threshold)
    print("Best F1:", best_f1)

    preds = (probs > best_threshold)

    sanity_checks(y_test, preds)
    leakage_check(X_train, X_test)

    report = {}

    report["baseline"] = baseline_sanity(
        model, X_test, y_test, X_train, y_train, probs, best_threshold
    )

    report["metrics"] = advanced_metrics(
        y_test, probs, best_threshold
    )

    report["robustness"] = robustness_tests(
        model, X_test, y_test, best_threshold
    )

    report["class_wise"] = class_wise_performance(y_test, preds)

    report["bootstrap_f1"] = bootstrap_f1(y_test, preds)
    
    report["group_kfold"] = group_kfold_cv(model_path, df)  

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
        "src_bytes",
        "dst_bytes",
        "src_ip_bytes",
        "dst_ip_bytes",
        "http_version",
        "http_method",
        "ssl_resumed",
        "label",
        "type",
    ], errors="ignore")

    y = df["label"].values
    X = X.values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train_raw, y_test_raw = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    X_train, y_train = create_sequences(X_train_scaled, y_train_raw, g_train)
    X_test, y_test   = create_sequences(X_test_scaled, y_test_raw, g_test)

    print(f"\n[DATA CHECK] Duplicates: {df.duplicated().sum()}")

    diagnose_model(args.model, df, X_test, y_test, X_train, y_train)