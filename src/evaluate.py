import torch
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score


def evaluate_model(model, X_test, y_test):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(X_test)
        outputs = torch.sigmoid(logits)
        preds = (outputs > 0.5).int().cpu().numpy()
    
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, outputs))


