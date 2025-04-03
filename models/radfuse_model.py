import torch
import torch.nn as nn
from models.vit_encoder import VisionEncoder
from models.clinicalbert_encoder import TextEncoder
from models.attention_fusion import CrossAttentionFusion

class RadFuse(nn.Module):
    def __init__(self, embed_dim=768, num_classes=14):
        super(RadFuse, self).__init__()
        self.image_encoder = VisionEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, images, input_ids, attention_mask):
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(input_ids, attention_mask)
        fused_feat = self.fusion(img_feat, txt_feat)
        return self.classifier(fused_feat)

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)
