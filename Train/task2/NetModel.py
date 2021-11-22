# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         NetModel
# Description:  
# Author:       Laity
# Date:         2021/11/10
# ---------------------------------------------
import torch.nn as nn
import torch
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
        x = self.fc3(F.relu(x))
        x = x.reshape(-1, 5)
        return x

#正确写法
class TextCnn(nn.Module):
    def __init__(self, word_bag_size):
        super().__init__()
        self.em_size = 256
        self.em = nn.Embedding(word_bag_size, self.em_size)
        self.conv = nn.Sequential(
            # 256 * 256 * 28
            nn.Conv1d(in_channels=self.em_size, out_channels=100, kernel_size=(5,)),
            # 256 * 100 * 24
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=24)
            # 256 * 100 * 1
        )
        self.fc1 = nn.Linear(100, 32)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.em(x)
        # Conv1d在列上做卷积 故要将维度交换 使列为词向量
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.reshape(256, -1)
        x = self.fc2(self.fc1(x))
        return x

# 错误TextCnn
class CNN(nn.Module):
    def __init__(self, word_bag_size):
        super().__init__()
        self.em_size = 256
        self.em = nn.Embedding(word_bag_size, self.em_size)
        # input: (batch_size, 28, hidden)
        self.cnn = nn.Conv1d(1, 1, (1, 5))
        self.fc1 = nn.Linear(63*28, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.em(x)
        # print(x.shape)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.squeeze(1)
        x = F.max_pool1d(x, 4)
        x = x.reshape(256, -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x