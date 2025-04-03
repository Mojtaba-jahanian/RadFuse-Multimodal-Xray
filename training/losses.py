import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelBCELoss(nn.Module):
    def __init__(self):
        super(MultiLabelBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        return self.bce(preds, targets)

class ContrastiveInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeds, text_embeds):
        logits = torch.matmul(image_embeds, text_embeds.T) / self.temperature
        targets = torch.arange(len(image_embeds)).to(image_embeds.device)
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        return (loss_i2t + loss_t2i) / 2
