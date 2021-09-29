# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/9/29
# ---------------------------------------------
import torch
import torch
from torch.nn import Linear, Module, MSELoss, Sequential
from torch.optim import SGD
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

# torch.save(rnn.state_dict(), 'rnn.pt')    保存
m_state_dict = torch.load('linear.pt')
new_m = Net()
new_m.load_state_dict(m_state_dict)
[w, b] = new_m.parameters()
w = w.item()
b = b.item()
print(w, b)