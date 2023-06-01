# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
# from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from transformers import DistilBertPreTrainedModel, DistilBertModel, DistilBertTokenizer
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F



class Distilbert(DistilBertPreTrainedModel):
    
    def __init__(self, config):
        super(Distilbert, self).__init__(config)
        self.distilbert = DistilBertModel(config)
        # self.pool_layer = BertPooler(config)
        self.hidden_size = config.hidden_size   #768
        self.num_classes = config.num_labels
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = nn.Tanh()


    def forward(self, input_ids, attention_mask, label=None):   
        output = self.distilbert(input_ids, attention_mask=attention_mask)
        # out = self.fc(output.pooler_output)
        # pooling
        first_token_tensor = output.last_hidden_state[:, 0]
        pooler_output = self.dense(first_token_tensor)
        # pooler_output = self.activation(pooler_output)
        # pooler_output = self.pool_layer(pooler_output)
        # class
        output = self.fc(pooler_output)
        return [output,pooler_output]


# class BertPooler(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.activation = nn.Tanh()

#     def forward(self, hidden_states):
#         # We "pool" the model by simply taking the hidden state corresponding
#         # to the first token.
#         first_token_tensor = hidden_states[:, 0]
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
#         return pooled_output