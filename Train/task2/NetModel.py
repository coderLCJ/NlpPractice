# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         NetModel
# Description:  
# Author:       Laity
# Date:         2021/11/10
# ---------------------------------------------
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, word_bag_size):
        super().__init__()
        self.em = nn.Embedding(word_bag_size, 256)
        # input: (batch_size, 28, 256)
        self.rnn = nn.RNN(256, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 5)