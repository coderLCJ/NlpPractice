# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         titanic
# Description:  
# Author:       Laity
# Date:         2021/10/14
# ---------------------------------------------
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def getData():
    data = []
    label = []
    sex = {'male': 1, 'female': 0, '': -1}
    embarked = {'C':0, 'Q':1, 'S': 2, '': -1}
    Cabin = {}
    Ticket = {}
    '''
       0           1      2       3   4   5    6     7    8     9    10     11
    PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    
    data:   Pclass Sex Age SibSp Parch Fare Embarked Ticket Cabin
    label: Survived(0/1)
    '''
    file = csv.reader(open('titanic/train.csv', 'r'))
    next(file)
    for line in file:
        info = [line[2], sex[line[4]], 0 if not line[5].isdigit() else float(line[5]), line[6], line[7], line[9], embarked[line[11]]]
        if line[8] not in Ticket:
            Ticket[line[8]] = len(Ticket)
        info.append(Ticket[line[8]])
        if line[10] not in Cabin:
            Cabin[line[10]] = len(Cabin)
        info.append(Cabin[line[10]])
        t = [float(i) for i in info]
        data.append(t)
        label.append(float(line[1]))
    return torch.tensor(data), torch.tensor(label)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 5)
        self.fc2 = nn.Linear(5, 2)
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 891
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())
data, label = getData()

valid_data, valid_label = data[500:], label[500:]
train_data, train_label = data[:500], label[:500]



for epoch in range(5000):

    optimizer.zero_grad()
    output = net(train_data)
    train_label = train_label.resize(500, 1)
    loss = criterion(output, train_label)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(loss)
        print(train_label[:10])
        print(output[:10])
