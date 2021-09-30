# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         Demo_0
# Description:  线性回归实践
# Author:       Laity
# Date:         2021/9/30
# ---------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_data():
    f = open('data/german.data-numeric')
    data = torch.zeros(1000, 24)
    label = torch.zeros(1000, 1)
    for i, line in enumerate(f.readlines()):
        line = line.split()
        d = [float(d) for d in line[:24]]
        data[i] = torch.tensor(d)
        label[i] = torch.tensor(float(line[-1]))
    # print(data)
    train_data, train_label = data[:900], label[:900]
    test_data, test_label = data[900:], label[900:]
    return train_data, train_label, test_data, test_label

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(24, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x

