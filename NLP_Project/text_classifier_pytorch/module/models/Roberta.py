# coding: UTF-8
import torch
import torch.nn as nn
from transformers import FlaxRobertaPreTrainedModel, RobertaModel, BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss


# class Roberta(FlaxRobertaPreTrainedModel):
class Roberta(BertPreTrainedModel):
    
    def __init__(self, config):
        super(Roberta, self).__init__(config)
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_labels
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input_ids, attention_mask, label=None):
  
        output = self.bert(input_ids, attention_mask=attention_mask)
        out = self.fc(output.pooler_output)
        return [out,output.pooler_output]

