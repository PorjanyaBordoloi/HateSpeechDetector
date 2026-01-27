import torch.nn as nn
from transformers import AutoModel

class MultilingualHateModel(nn.Module):
    def __init__(self, model_name='xlm-roberta-base'):
        super(MultilingualHateModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.classifier(output)