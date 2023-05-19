# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         d2l_RNN
# Description:  
# Author:       Laity
# Date:         2021/10/25
# ---------------------------------------------
import torch
from d2l import torch as d2l
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

batch_size, num_step = 32, 35
hidden_size = 256

# train_iter: 2-gram [0]:train [1]:label
# vocab: 所有字符
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_step)


class rnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = len(vocab)
        self.rnn = nn.RNN(self.vocab_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, x, h_state):

        x, h_state = self.rnn(x, h_state)
        out = self.fc(x)
        return out, h_state




net = rnnNet()
h_state = None

x = torch.randn((batch_size, num_step, len(vocab)))
y, h_state = net(x, h_state)
print(y.shape)
print(h_state.shape)