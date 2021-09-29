# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/9/29
# ---------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import getData

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 这里论文上写的是conv,官方教程用了线性层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(x, (2, 2))
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # print(x.size())
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print(x.size())
        x = x.view(-1, self.num_flat_features(x)) # 此步必须有
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
epochs = 100
batch_size = 40
train_data, train_label = getData.getTrainData()

for epoch in range(epochs):
    acc = 0.0
    los = 0.0
    inputs, target = train_data[epoch * batch_size: (epoch + 1) * batch_size], train_label[
        epoch * batch_size: (epoch + 1) * batch_size]
    for i in range(batch_size):
        net.zero_grad()
        outputs = net(inputs[i])
        loss = criterion(outputs, target[i].long())
        loss.backward()
        optimizer.step()
        los += loss
        predict = 0
        for j in range(15):
            if outputs[0][j] == max(outputs[0]):
                predict = j
                break
        # print(predict, ' === ', train_label[i])
        if predict == train_label[i]:
            acc += 1

    print('epoch: %d acc: %.2f%% loss: %.3f' % (epoch, acc/batch_size, los))
