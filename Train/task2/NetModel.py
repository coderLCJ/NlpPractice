# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         NetModel
# Description:  
# Author:       Laity
# Date:         2021/11/10
# ---------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, word_bag_size):
        super().__init__()
        # hidden_size = 28,  ac =
        # hidden_size = 256, ac =
        self.em_size = 256
        self.em = nn.Embedding(word_bag_size, self.em_size)
        # input: (batch_size, 28, hidden)
        self.rnn = nn.LSTM(28 * self.em_size, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.em(x)
        x = x.reshape(-1, 1, 28*self.em_size)
        x, _ = self.rnn(x)
        x = self.fc1(F.relu(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.reshape(-1, 5)
        return x

