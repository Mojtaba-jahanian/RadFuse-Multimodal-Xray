import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def evaluate_classification(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            images, reports, labels = batch
            images = images.to(device)
            reports = reports.to(device)
            labels = labels.to(device)

            logits = model(images, reports)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)

    auc = roc_auc_score(y_true, y_pred, average='macro')
    y_pred_bin = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_bin, average='macro')

    return {'AUC': auc, 'F1': f1}
