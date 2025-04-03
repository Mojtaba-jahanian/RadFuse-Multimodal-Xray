import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.radfuse_model import RadFuseModel
from data.mimiccxr_preprocessing import MIMICDataset
from training.losses import MultiLabelBCELoss, ContrastiveInfoNCELoss
from training.utils import set_seed, save_checkpoint

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(config):

    # Set seed for reproducibility
    set_seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and Dataloader
    train_dataset = MIMICDataset(
        image_dir=config['data']['image_path'],
        report_dir=config['data']['report_path'],
        mode='train'
    )
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Model
    model = RadFuseModel(config['model']).to(device)

    # Losses
    classification_loss = MultiLabelBCELoss()
    contrastive_loss = ContrastiveInfoNCELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    # Training Loop
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_cls_loss = 0
        epoch_ctr_loss = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            reports = batch['report'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits, img_embeds, txt_embeds = model(images, reports)

            loss_cls = classification_loss(logits, labels)
            loss_ctr = contrastive_loss(img_embeds, txt_embeds)
            loss = loss_cls + loss_ctr

            loss.backward()
            optimizer.step()

            epoch_cls_loss += loss_cls.item()
            epoch_ctr_loss += loss_ctr.item()

        print(f"[Epoch {epoch+1}] Classification Loss: {epoch_cls_loss:.4f} | Contrastive Loss: {epoch_ctr_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, f"checkpoints/radfuse_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
