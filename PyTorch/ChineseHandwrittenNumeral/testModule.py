# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         testModule
# Description:  
# Author:       Laity
# Date:         2021/9/29
# ---------------------------------------------
import torch
import main
import getData
import matplotlib.pyplot as plt

net = main.Net()
dic = torch.load('net.pt')
net.load_state_dict(dic)
testData, testLabel = getData.getTestData()
cha = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']

acc = 0
for i in range(getData.test_size): # len(testLabel)
    outputs = net(testData[i])
    res = 0
    for j in range(15):
        if outputs[0][j] == max(outputs[0]):
            res = j
            break
    print('predict = %s   right = %s' % (cha[res], cha[int(testLabel[i].item())]))
    if res == int(testLabel[i].item()):
        acc += 1
    plt.imshow(testData[i].reshape(64, 64))
    plt.show()

print('acc = %d/%d' % (acc, getData.test_size))