# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         handTest
# Description:  
# Author:       Laity
# Date:         2021/9/29
# ---------------------------------------------
import os
import cv2
import matplotlib.pyplot as plt
import torch
import main

def getHandData():
    data_name = os.listdir('handwrite')
    print(data_name)
    for name in data_name:
        if name == 'data':
            break
        old = cv2.imread('handwrite/' + name)
        new = cv2.resize(old, (64, 64))
        cv2.imwrite('handwrite/data/' + name, new)



img = plt.imread('t.jpg')
print(len(img))
plt.imshow(img)
plt.show()






# def getData():
#     testData = torch.zeros(6, 1, 1, 64, 64)
#     testLabel = torch.zeros(6, 1)
#     data_name = os.listdir('handwrite/data')
#     for i, name in enumerate(data_name):
#         img = plt.imread('handwrite/data/' + name)
#         testData[i] = torch.tensor(img).sum(dim=2)
#         testLabel[i] = torch.tensor(int(name[0]))
#
#     return testData, testLabel


# getHandData()
# testData, testLabel = getData()
# net = main.Net()
# cha = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']
# for i in range(20): # len(testLabel)
#     outputs = net(testData[i])
#     res = 0
#     for j in range(6):
#         if outputs[0][j] == max(outputs[0]):
#             res = j
#             break
#     print('predict = %s   right = %s' % (cha[res], cha[int(testLabel[i].item())+1]))
#     plt.imshow(testData[i].reshape(64, 64))
#     plt.show()