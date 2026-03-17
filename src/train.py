import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

class LSTMModel(nn.Module):

    def __init__(self, input_size):

        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


def train_model(X_train, y_train, epochs=20, batch_size=512):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_size=X_train.shape[2]).to(device)

    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(epochs):
        total_loss = 0

        for xb, yb in loader:
            optimizer.zero_grad()

            outputs = model(xb).squeeze()

            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader)}")

    joblib.dump(model, "../model/lstm_model.pkl")

    return model