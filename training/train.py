# radfuse/train.py
import os
import torch
from torch.utils.data import DataLoader
from radfuse.data.dataset import RadFuseDataset
from radfuse.model.model import RadFuseModel
from radfuse.utils.utils import set_seed
from transformers import AdamW


def train(args):
    set_seed(args.seed)

    print("[INFO] Loading dataset...")
    train_set = RadFuseDataset(args.data_dir, split='train')
    val_set = RadFuseDataset(args.data_dir, split='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    print("[INFO] Initializing model...")
    model = RadFuseModel()
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn_cls = torch.nn.BCELoss()
    best_auc = 0.0

    print("[INFO] Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images, reports, labels = batch
            images, reports, labels = images.to(args.device), reports.to(args.device), labels.to(args.device)
            logits, _ = model(images, reports)
            loss = loss_fn_cls(logits, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}")

        # validation
        model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            for batch in val_loader:
                images, reports, labels = batch
                images, reports = images.to(args.device), reports.to(args.device)
                logits, _ = model(images, reports)
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels)

        # you can integrate sklearn.metrics.roc_auc_score, etc.
        print(f"[INFO] Finished Epoch {epoch+1}")

    print("[INFO] Training completed.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train(args)
