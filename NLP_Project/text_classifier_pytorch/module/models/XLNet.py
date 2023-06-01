# coding: UTF-8
import torch
import torch.nn as nn
from transformers import XLNetPreTrainedModel, XLNetModel, AutoModel


class XLNet(XLNetPreTrainedModel):
    
    def __init__(self, config):
        super(XLNet, self).__init__(config)
        self.xlnet = AutoModel.from_config(config)
        # self.xlnet = XLNetModel(config)
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_labels
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = nn.Tanh()
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input_ids, attention_mask, label=None):   
        output = self.xlnet(input_ids, attention_mask=attention_mask)
        # pooling
        first_token_tensor = output.last_hidden_state[:, 0]
        pooler_output = self.dense(first_token_tensor)
        # pooler_output = self.activation(pooler_output)
        out = self.fc(pooler_output)
        return [out,pooler_output]

