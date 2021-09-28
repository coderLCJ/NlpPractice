# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         linear
# Description:  
# Author:       Laity
# Date:         2021/9/28
# ---------------------------------------------
import torch
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# x = np.linspace(0,2,50)
# y = 5 * x + 7
# plt.plot(x, y)

def createData():
    x = torch.rand(256)
    noise = torch.randn(256) / 4
    y = x * 5 + 7 + noise
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    # sns.lmplot(x='x', y='y', data=df, fit_reg=False)  # fit_reg：是否进行拟合
    plt.scatter(x, y)
    # plt.show()
    return x, y


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x


data, label = createData()
epochs = 100
criterion = MSELoss()
net = Net()
optimizer = SGD(net.parameters(), lr=0.1)

for epoch in range(epochs):
    inputs = data.reshape(-1, 1)
    labels = label.reshape(-1, 1)
    net.zero_grad()
    outputs = net(inputs)
    # print('outputs = ', outputs.reshape(1, -1)[0][:20])
    # print('labels = ', labels.reshape(1, -1)[0][:20])
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    # if epoch % 10 == 0:
    # print('loss = %.3f\n' % loss)

[w, b] = net.parameters()
w = w.item()
b = b.item()
x = torch.rand(256)
y = w * x + b
plt.plot(x, y)
plt.show()
