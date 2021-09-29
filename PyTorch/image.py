# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         image
# Description:  
# Author:       Laity
# Date:         2021/9/27
# ---------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Animator import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 展示图像的函数
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print(np.transpose(npimg, (1, 2, 0)))
    print(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# # 获取随机数据
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # 显示图像标签
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# # 展示图像
# imshow(torchvision.utils.make_grid(images))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
animator = Animator(xlabel='epoch', xlim=[1, 10], ylim=[0, 2],
                    legend=['loss', 'acc'])
# animator.add(epoch + 1, train_metrics + (test_acc,))
animator.add(0, (2, 0))
for epoch in range(10):  # 多批次循环
    running_loss = 0.0
    acc = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data
        # 梯度置0
        optimizer.zero_grad()
        # print(inputs[0], '\ntp = ', type(inputs))     # 输入数据 tensor
        # print(labels[0], '\ntp = ', type(labels))       # 标签 tensor
        exit()
        # 正向传播，反向传播，优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        for j, pre in enumerate(outputs):
            temp = list(pre)
            if temp.index(max(temp)) == labels[j]:
                acc += 1
        # 打印状态信息
        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            print('[%d, %5d] acc: %.3f%%' %
                  (epoch + 1, i + 1, acc / 8000 * 100))
            animator.add(epoch + 1, (running_loss / 2000, acc / 8000))
            running_loss = 0.0
            acc = 0.0
print('Finished Training')
