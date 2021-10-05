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
from d2l import torch

def get_data():
    f = open('data/german.data-numeric')
    data = torch.zeros(1000, 24)
    label = torch.zeros(1000)
    for i, line in enumerate(f.readlines()):
        line = line.split()
        d = [float(d) for d in line[:24]]
        data[i] = torch.tensor(d)
        label[i] = torch.tensor(float(line[-1]) - 1)
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


trainData, trainLabel, testData, testLabel = get_data()
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

acc = 0.0
loss_ = 0.0
s = 0.0
for epoch in range(6000):
    net.zero_grad()
    outputs = net(trainData)
    loss = criterion(outputs, trainLabel.long())
    loss.backward()
    optimizer.step()
    loss_ += loss
    for i in range(900):
        if outputs[i][0] > outputs[i][1] and trainLabel[i] == 0 or outputs[i][1] > outputs[i][0] and trainLabel[i] == 1:
            acc += 1
        s += 1
    if epoch % 100 == 0:
        # print('loss: %.5f   acc: %.5f%%' % (loss/100, acc / s * 100))
        acc = 0
        s = 0
        loss_ = 0
        predict = net(testData)
        right = 0
        for j in range(100):
            if (predict[j][0] > predict[j][1] and testLabel[j] == 0) or (predict[j][0] < predict[j][1] and testLabel[j] == 1):
                right += 1
        print('acc: %d' % right)
