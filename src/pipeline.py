import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import Config


@dataclass
class SequenceDatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    input_dim: int
    preprocessor: object
    metadata: dict


def load_dataset(config: Config) -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PATH)

    required = (
        [config.TARGET_COL, config.DEVICE_COL]
        + config.NUMERIC_COLS
        + config.CATEGORICAL_COLS
    )

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    return df


def prepare_row_order(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = df.copy()
    df["_original_order"] = np.arange(len(df))

    if config.OPTIONAL_TIME_COL and config.OPTIONAL_TIME_COL in df.columns:
        df = df.sort_values(
            by=[config.DEVICE_COL, config.OPTIONAL_TIME_COL, "_original_order"]
        ).reset_index(drop=True)
    else:
        df = df.sort_values(
            by=[config.DEVICE_COL, "_original_order"]
        ).reset_index(drop=True)

    return df


def split_devices_or_rows(df: pd.DataFrame, config: Config):
    if config.SPLIT_BY_DEVICE:
        device_ids = df[config.DEVICE_COL].dropna().astype(str).unique()

        train_devices, test_devices = train_test_split(
            device_ids,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            shuffle=True,
        )

        relative_val_size = config.VAL_SIZE / (1.0 - config.TEST_SIZE)

        train_devices, val_devices = train_test_split(
            train_devices,
            test_size=relative_val_size,
            random_state=config.RANDOM_STATE,
            shuffle=True,
        )

        train_df = df[df[config.DEVICE_COL].astype(str).isin(train_devices)].copy()
        val_df = df[df[config.DEVICE_COL].astype(str).isin(val_devices)].copy()
        test_df = df[df[config.DEVICE_COL].astype(str).isin(test_devices)].copy()

        split_info = {
            "split_mode": "device",
            "n_train_devices": len(train_devices),
            "n_val_devices": len(val_devices),
            "n_test_devices": len(test_devices),
        }
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=df[config.TARGET_COL],
        )

        relative_val_size = config.VAL_SIZE / (1.0 - config.TEST_SIZE)

        train_df, val_df = train_test_split(
            train_df,
            test_size=relative_val_size,
            random_state=config.RANDOM_STATE,
            stratify=train_df[config.TARGET_COL],
        )

        split_info = {
            "split_mode": "row",
        }

    return train_df, val_df, test_df, split_info


def build_preprocessor(config: Config):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_COLS),
            ("cat", categorical_pipeline, config.CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    return preprocessor


def fit_transform_rows(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: Config):
    preprocessor = build_preprocessor(config)

    feature_cols = config.NUMERIC_COLS + config.CATEGORICAL_COLS

    X_train_rows = preprocessor.fit_transform(train_df[feature_cols])
    X_val_rows = preprocessor.transform(val_df[feature_cols])
    X_test_rows = preprocessor.transform(test_df[feature_cols])

    y_train_rows = train_df[config.TARGET_COL].astype(int).values
    y_val_rows = val_df[config.TARGET_COL].astype(int).values
    y_test_rows = test_df[config.TARGET_COL].astype(int).values

    return (
        X_train_rows,
        y_train_rows,
        X_val_rows,
        y_val_rows,
        X_test_rows,
        y_test_rows,
        preprocessor,
    )


def assign_sequence_label(window_labels: np.ndarray, mode: str) -> int:
    if mode == "last":
        return int(window_labels[-1])
    if mode == "majority":
        return int(np.mean(window_labels) >= 0.5)
    if mode == "any_attack":
        return int(np.any(window_labels == 1))
    raise ValueError(f"Unknown SEQUENCE_LABEL_MODE: {mode}")


def build_sequences_from_rows(
    X_rows: np.ndarray,
    y_rows: np.ndarray,
    meta_df: pd.DataFrame,
    config: Config,
):
    sequences = []
    seq_labels = []
    seq_meta = []

    seq_len = config.SEQUENCE_LENGTH
    stride = config.STRIDE

    start_idx = 0
    grouped = meta_df.groupby(config.DEVICE_COL, sort=False)

    for device_id, group in grouped:
        group_indices = group.index.to_numpy()
        device_X = X_rows[group_indices]
        device_y = y_rows[group_indices]

        if len(group_indices) < max(config.MIN_ROWS_PER_DEVICE, seq_len):
            continue

        max_start = len(group_indices) - seq_len
        for s in range(0, max_start + 1, stride):
            e = s + seq_len

            window_X = device_X[s:e]
            window_y = device_y[s:e]

            if len(window_X) < seq_len:
                if config.DROP_INCOMPLETE_WINDOWS:
                    continue

            label = assign_sequence_label(window_y, config.SEQUENCE_LABEL_MODE)

            sequences.append(window_X.astype(np.float32))
            seq_labels.append(label)
            seq_meta.append(
                {
                    "device_id": str(device_id),
                    "start_pos": int(s),
                    "end_pos": int(e - 1),
                }
            )

    if len(sequences) == 0:
        raise ValueError(
            "No sequences were created. Check sequence length, device grouping, or split settings."
        )

    X_seq = np.stack(sequences)
    y_seq = np.array(seq_labels, dtype=np.float32)

    return X_seq, y_seq, seq_meta


def create_sequence_bundle(config: Config) -> SequenceDatasetBundle:
    df = load_dataset(config)
    df = prepare_row_order(df, config)

    drop_cols = [col for col in config.ANALYSIS_ONLY_COLS if col in df.columns]
    df_model = df.drop(columns=drop_cols, errors="ignore").copy()

    train_df, val_df, test_df, split_info = split_devices_or_rows(df_model, config)

    (
        X_train_rows,
        y_train_rows,
        X_val_rows,
        y_val_rows,
        X_test_rows,
        y_test_rows,
        preprocessor,
    ) = fit_transform_rows(train_df, val_df, test_df, config)

    # important: preserve row alignment
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train_seq, y_train_seq, train_meta = build_sequences_from_rows(
        X_train_rows, y_train_rows, train_df, config
    )
    X_val_seq, y_val_seq, val_meta = build_sequences_from_rows(
        X_val_rows, y_val_rows, val_df, config
    )
    X_test_seq, y_test_seq, test_meta = build_sequences_from_rows(
        X_test_rows, y_test_rows, test_df, config
    )

    metadata = {
        "split_info": split_info,
        "train_sequences": len(X_train_seq),
        "val_sequences": len(X_val_seq),
        "test_sequences": len(X_test_seq),
        "sequence_length": config.SEQUENCE_LENGTH,
        "input_dim": X_train_seq.shape[-1],
        "train_positive_ratio": float(y_train_seq.mean()),
        "val_positive_ratio": float(y_val_seq.mean()),
        "test_positive_ratio": float(y_test_seq.mean()),
        "train_seq_meta_sample": train_meta[:5],
        "val_seq_meta_sample": val_meta[:5],
        "test_seq_meta_sample": test_meta[:5],
    }

    with open(config.PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(preprocessor, f)

    return SequenceDatasetBundle(
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_val=X_val_seq,
        y_val=y_val_seq,
        X_test=X_test_seq,
        y_test=y_test_seq,
        input_dim=X_train_seq.shape[-1],
        preprocessor=preprocessor,
        metadata=metadata,
    )