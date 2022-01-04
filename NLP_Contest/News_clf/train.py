# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         train
# Description:  
# Author:       Laity
# Date:         2022/1/4
# ---------------------------------------------
import torch.cuda
from data_process import load_data as load_data
from model import Net as Net
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

devices = 'cuda' if torch.cuda.is_available() else 'cpu'

net = Net().to(devices)
data = load_data()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

t = 0
for x, y in data:
    predict = net(x.to(devices))
    loss = criterion(predict.to(devices), y.to(devices))
    net.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)
    if t == 20:
        break