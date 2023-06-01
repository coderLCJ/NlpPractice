
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from torch.nn import CrossEntropyLoss


class Bert(BertPreTrainedModel):
    
    def __init__(self, config):
        super(Bert, self).__init__(config)
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_labels
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, 
                input_ids, 
                attention_mask, 
                label=None, 
                input_ids_anti=None, 
                label_anti=None):
        # inference  
        output_bert = self.bert(input_ids, attention_mask=attention_mask)    #(batch_size, sen_length, hidden_size)
        output_pooler = output_bert.pooler_output
        output = self.fc(output_pooler)
        
        return [output, output_pooler]

