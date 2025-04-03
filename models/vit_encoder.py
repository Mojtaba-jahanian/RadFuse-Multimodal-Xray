import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super(VisionEncoder, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        self.projector = nn.Linear(768, embed_dim)

    def forward(self, x):
        features = self.vit(x)
        return self.projector(features)
