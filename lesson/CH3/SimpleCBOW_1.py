# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         SimpleCBOW_1
# Description:  引入embedding层
# Author:       Laity
# Date:         2021/10/11
# ---------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

data = 'you say hello and i say goodbye .'


def getData():
    content = []
    target = []
    words = data.split()
    leng = len(words)
    for i in range(leng):
        if 0 < i < leng-1:
            content.append((words[i-1], words[i+1]))
            target.append(words[i])
    word2OneHot = {}
    t = 0
    for i, word in enumerate(words):
        if word not in words[0:t]:
            words[t] = word
            t += 1
    words = words[:t]
    size = len(words)

    for word in words:
        if word not in word2OneHot:
            word2OneHot[word] = np.zeros(size)
            word2OneHot[word][words.index(word)] = 1

    trainData = np.zeros((leng-2, 2, size))
    trainLabel = np.zeros((leng-2, size))
    for i in range(leng-2):
        trainData[i][0] = word2OneHot[content[i][0]]
        trainData[i][1] = word2OneHot[content[i][1]]
        trainLabel[i][words.index(target[i])] = 1

    return trainData, trainLabel, words

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.eb = nn.Embedding(7, 2)
        self.fc0 = nn.Linear(2, 3)
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 7)

    def forward(self, x1, x2):
        x1 = self.eb(x1.max(-1)[1])
        x2 = self.eb(x2.max(-1)[1])
        x1 = self.fc0(x1)
        x2 = self.fc1(x2)
        y = (x1 + x2) * 0.5
        y = self.fc2(y)
        return y


_, __, words = getData()
inputs = torch.tensor(_, dtype=torch.float32)
target = torch.tensor(__, dtype=torch.float32)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(1000):
    x1 = inputs[:, 0]
    x2 = inputs[:, 1]
    y = target.max(-1)[1]
    y_hat = net(x1, x2)
    net.zero_grad()
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 < 10:
        for i in range(6):
            w = words[y_hat[i].max(-1)[1]]
            for j in range(6):
                if j != y[i]:
                    print(words[j], end=' ')
                else:
                    print('?', end=' ')
            print('\npredict = %s, true = %s' % (w, words[y[i]]))
            print('[%s] [%s]' % (y_hat, y))
        print(loss)
