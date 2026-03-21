from pathlib import Path


class Config:
    # =========================================================
    # Paths
    # =========================================================
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = PROJECT_ROOT.parent / "Data" / "iot_dataset_clean.csv"   # adjust if needed
    MODEL_DIR = PROJECT_ROOT / "model"
    REPORTS_DIR = PROJECT_ROOT / "reports"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Dataset columns
    # =========================================================
    TARGET_COL = "label"
    DEVICE_COL = "src_ip"
    ANALYSIS_ONLY_COLS = ["type"]   # not used in training
    OPTIONAL_TIME_COL = None        # example: "ts" if you later have one

    NUMERIC_COLS = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "missed_bytes",
        "src_pkts",
        "dst_pkts",
        "src_ip_bytes",
        "dst_ip_bytes",
        "dns_qclass",
        "dns_qtype",
        "http_trans_depth",
        "http_request_body_len",
        "http_response_body_len",
    ]

    CATEGORICAL_COLS = [
        "proto",
        "ssl_version",
        "ssl_cipher",
        "ssl_resumed",
        "http_method",
        "http_version",
    ]

    # =========================================================
    # Sequence settings
    # =========================================================
    SEQUENCE_LENGTH = 20
    STRIDE = 1

    # Sequence label strategy:
    # "last"      -> use label of last row in window
    # "majority"  -> majority vote in window
    # "any_attack"-> 1 if any row in window is attack
    SEQUENCE_LABEL_MODE = "last"

    DROP_INCOMPLETE_WINDOWS = True
    MIN_ROWS_PER_DEVICE = 20

    # =========================================================
    # Split settings
    # =========================================================
    RANDOM_STATE = 42
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15   # applied on remaining after test split

    # IMPORTANT:
    # split by device to avoid leakage between train/val/test
    SPLIT_BY_DEVICE = True

    # =========================================================
    # Training settings
    # =========================================================
    DEVICE = "cuda"   # automatically handled if unavailable
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # =========================================================
    # Model settings
    # =========================================================
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = False

    # =========================================================
    # Early stopping
    # =========================================================
    EARLY_STOPPING_PATIENCE = 5

    # =========================================================
    # Saved files
    # =========================================================
    MODEL_PATH = MODEL_DIR / "best_lstm_model.pt"
    PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
    METRICS_PATH = REPORTS_DIR / "metrics.json"
    DIAGNOSIS_PATH = REPORTS_DIR / "diagnosis.txt"