# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         model
# Description:  
# Author:       Laity
# Date:         2022/3/28
# ---------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self, word_size):
        super().__init__()
        # x = (batch_size, seq_size)
        self.em = nn.Embedding(word_size, 64)
        self.rnn = nn.LSTM(input_size=128, hidden_size=256, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.Sigmoid(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        # 批次一定要自动计算
        # batch, seq_num, seq_len = 64, 2, 32
        x = self.em(x)
        # 64, 2, 32, 64
        # print(x.shape)
        x = x.reshape(-1, 32, 128)
        x, (h, c) = self.rnn(x)
        h = h.reshape(-1, 512)
        x = self.fc(h)
        # print(x.shape)
        return x


if __name__ == '__main__':
    t = np.random.randint(0, 100, 32 * 64 * 2)
    t = torch.LongTensor(t)
    t = t.reshape(64, 2, 32)
    print(t.shape)
    net = Net(100)
    net(t)