import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", embed_dim=768):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.projector = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.projector(cls_token)
