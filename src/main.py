import json

from config import Config
from diagnose import generate_diagnosis_report
from evaluate import evaluate_model
from pipeline import create_sequence_bundle
from train import train_model


def main():
    config = Config()

    print("Creating sequence bundle...")
    bundle = create_sequence_bundle(config)

    print("Generating diagnosis report...")
    diagnosis = generate_diagnosis_report(bundle, config)
    print(diagnosis)

    print("Training model...")
    model, history = train_model(bundle, config)

    print("Evaluating on validation set...")
    val_metrics = evaluate_model(model, bundle.X_val, bundle.y_val, config, split_name="validation")

    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, bundle.X_test, bundle.y_test, config, split_name="test")

    results = {
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "metadata": bundle.metadata,
    }

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nValidation Metrics:")
    for k, v in val_metrics.items():
        print(f"{k}: {v}")

    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    print(f"\nSaved model to: {config.MODEL_PATH}")
    print(f"Saved reports to: {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()