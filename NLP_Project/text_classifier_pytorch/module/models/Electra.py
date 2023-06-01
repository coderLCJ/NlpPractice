# coding: UTF-8
import torch
import torch.nn as nn
from transformers import ElectraPreTrainedModel, ElectraModel, ElectraTokenizer
from torch.nn import CrossEntropyLoss


class Electra(ElectraPreTrainedModel):
    
    def __init__(self, config):
        super(Electra, self).__init__(config)
        self.electra = ElectraModel(config)
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_labels
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask, label=None):    
        output = self.electra(input_ids, attention_mask=attention_mask)
        
        first_token_tensor = output.last_hidden_state[:, 0]
        pooler_output = self.dense(first_token_tensor)
        pooler_output = self.activation(pooler_output)
        output = self.fc(pooler_output)
        return [output,pooler_output]

