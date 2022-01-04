# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         model
# Description:  
# Author:       Laity
# Date:         2022/1/4
# ---------------------------------------------
import torch
import torch.nn as nn


max_len = 58000
max_num = 7549

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.em = nn.Embedding(max_num, 2048)
        # 128 * 58000 * 1024
        self.TextCNN = nn.Sequential(
            nn.Conv1d(2048, 1024, (8,)),
            nn.Conv1d(1024, 512, (8,)),
            nn.MaxPool1d(16)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(3624, 1024),
            nn.Dropout(0.9),
            nn.Linear(1024, 512)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 14)
        )

    def forward(self, x):
        x = self.em(x)
        x = x.permute(0, 2, 1)
        x = self.TextCNN(x)
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.fc2(x)
        x = x.reshape(-1, 512)
        x = self.fc3(x)
        return x