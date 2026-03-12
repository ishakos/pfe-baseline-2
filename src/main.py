from sklearn.model_selection import train_test_split
from pipeline import load_dataset, scale_features, create_sequences
from train import train_model
from evaluate import evaluate_model


DATA_PATH = "../../Data/iot_dataset_clean.csv"


X, y = load_dataset(DATA_PATH)

X_scaled, scaler = scale_features(X)

X_seq, y_seq = create_sequences(X_scaled, y.values)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq,
    y_seq,
    test_size=0.2,
    random_state=42,
    stratify=y_seq
)

model = train_model(X_train, y_train)

evaluate_model(model, X_test, y_test)