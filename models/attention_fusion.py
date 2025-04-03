import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_feat, txt_feat):
        # Add sequence dimension
        img_feat = img_feat.unsqueeze(1)
        txt_feat = txt_feat.unsqueeze(1)

        # Self-attention on text
        txt_feat_attn, _ = self.self_attn(txt_feat, txt_feat, txt_feat)

        # Cross-attention: image attends to text
        fused_feat, _ = self.cross_attn(img_feat, txt_feat_attn, txt_feat_attn)

        # Residual and normalization
        output = self.norm(fused_feat + img_feat)

        return output.squeeze(1)
