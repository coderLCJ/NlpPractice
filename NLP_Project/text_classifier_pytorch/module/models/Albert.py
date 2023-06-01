# coding: UTF-8
import torch
import torch.nn as nn
from transformers import AlbertPreTrainedModel, AlbertModel
from torch.nn import CrossEntropyLoss


class Albert(AlbertPreTrainedModel):
    
    def __init__(self, config):
        super(Albert, self).__init__(config)
        self.albert = AlbertModel(config)
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_labels
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input_ids, attention_mask, label=None):  
        output_albert = self.albert(input_ids, attention_mask=attention_mask)
        output = self.fc(output_albert.pooler_output)
        return [output,output_albert.pooler_output]

