from collections import Counter

import numpy as np

from config import Config


def generate_diagnosis_report(bundle, config: Config):
    lines = []
    lines.append("LSTM Sequence Diagnosis Report")
    lines.append("=" * 50)
    lines.append("")

    lines.append("Split information:")
    for k, v in bundle.metadata["split_info"].items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("Sequence information:")
    lines.append(f"- sequence_length: {bundle.metadata['sequence_length']}")
    lines.append(f"- input_dim: {bundle.metadata['input_dim']}")
    lines.append(f"- train_sequences: {bundle.metadata['train_sequences']}")
    lines.append(f"- val_sequences: {bundle.metadata['val_sequences']}")
    lines.append(f"- test_sequences: {bundle.metadata['test_sequences']}")

    lines.append("")
    lines.append("Class ratios:")
    lines.append(f"- train positive ratio: {bundle.metadata['train_positive_ratio']:.4f}")
    lines.append(f"- val positive ratio: {bundle.metadata['val_positive_ratio']:.4f}")
    lines.append(f"- test positive ratio: {bundle.metadata['test_positive_ratio']:.4f}")

    lines.append("")
    lines.append("Sanity checks:")
    lines.append(f"- X_train shape: {bundle.X_train.shape}")
    lines.append(f"- X_val shape: {bundle.X_val.shape}")
    lines.append(f"- X_test shape: {bundle.X_test.shape}")
    lines.append(f"- y_train shape: {bundle.y_train.shape}")
    lines.append(f"- y_val shape: {bundle.y_val.shape}")
    lines.append(f"- y_test shape: {bundle.y_test.shape}")

    lines.append("")
    lines.append("Metadata samples:")
    lines.append(f"- train meta sample: {bundle.metadata['train_seq_meta_sample']}")
    lines.append(f"- val meta sample: {bundle.metadata['val_seq_meta_sample']}")
    lines.append(f"- test meta sample: {bundle.metadata['test_seq_meta_sample']}")

    report = "\n".join(lines)

    with open(config.DIAGNOSIS_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    return report