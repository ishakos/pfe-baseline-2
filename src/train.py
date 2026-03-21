import copy
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        output, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]

        x = self.dropout(last_hidden)
        logits = self.fc(x).squeeze(1)
        return logits


def get_device(config: Config):
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dataloaders(bundle, config: Config):
    X_train = torch.tensor(bundle.X_train, dtype=torch.float32)
    y_train = torch.tensor(bundle.y_train, dtype=torch.float32)

    X_val = torch.tensor(bundle.X_val, dtype=torch.float32)
    y_val = torch.tensor(bundle.y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


def binary_classification_metrics_from_logits(logits, y_true):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    y_true_np = y_true.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()

    tp = ((preds_np == 1) & (y_true_np == 1)).sum()
    tn = ((preds_np == 0) & (y_true_np == 0)).sum()
    fp = ((preds_np == 1) & (y_true_np == 0)).sum()
    fn = ((preds_np == 0) & (y_true_np == 1)).sum()

    accuracy = (tp + tn) / max(len(y_true_np), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_logits.append(logits)
            all_targets.append(y_batch)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = binary_classification_metrics_from_logits(all_logits, all_targets)
    avg_loss = total_loss / max(total_samples, 1)

    return avg_loss, metrics


def train_model(bundle, config: Config):
    device = get_device(config)

    model = LSTMClassifier(
        input_dim=bundle.input_dim,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
    ).to(device)

    train_loader, val_loader = make_dataloaders(bundle, config)

    pos_count = float(bundle.y_train.sum())
    neg_count = float(len(bundle.y_train) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_state = None
    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total_samples = 0

        all_train_logits = []
        all_train_targets = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_train_logits.append(logits.detach())
            all_train_targets.append(y_batch.detach())

        train_logits = torch.cat(all_train_logits, dim=0)
        train_targets = torch.cat(all_train_targets, dim=0)
        train_metrics = binary_classification_metrics_from_logits(train_logits, train_targets)
        train_loss = running_loss / max(total_samples, 1)

        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_f1"].append(float(train_metrics["f1"]))
        history["val_f1"].append(float(val_metrics["f1"]))

        print(
            f"Epoch {epoch:02d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best model state saved.")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), config.MODEL_PATH)

    return model, history