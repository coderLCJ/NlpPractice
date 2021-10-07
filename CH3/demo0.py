# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:
# Author:       Laity
# Date:         2021/10/7
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
        trainLabel[i] = words.index(target[i])

    return trainData, trainLabel

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(7, 2)

    def forward(self, x):
        x = self.fc(x)
        return x



_, __ = getData()
inputs = torch.tensor(_, dtype=torch.float32)
target = torch.tensor(__, dtype=torch.float32)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

print(inputs.shape)
print(target.shape)

for epoch in range(1):
    outputs = net(inputs)
    print(outputs)