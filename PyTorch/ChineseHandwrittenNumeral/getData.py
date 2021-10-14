# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         getData
# Description:  
# Author:       Laity
# Date:         2021/9/29
# ---------------------------------------------
import os

import numpy as np
import torch
from matplotlib import image as mpimage, pyplot as plt

train_size = 4000
test_size = 100
image_name = os.listdir('E:/DESKTOP/Github/DATA/data/RawDataset')

def randNum(begin, end):
    size = end - begin
    book = {}
    i = 0
    while i < size:
        pos = np.random.randint(begin, end)
        if pos not in book.keys():
            book[pos] = 1
            i += 1
            yield pos

def getTrainData():
    train_data = torch.zeros(train_size, 1, 1, 64, 64)
    train_label = torch.zeros(train_size, 1)
    for pos, i in enumerate(randNum(0, train_size)):
        img = mpimage.imread('E:/DESKTOP/Github/DATA/data/RawDataset/' + image_name[i])
        train_data[pos] = torch.tensor(img)
        train_label[pos] = (int(image_name[i].split(',')[2].split('}'[0][0])[0]))-1

    return train_data, train_label


def getTestData():
    test_data = torch.zeros(test_size, 1, 1, 64, 64)
    test_label = torch.zeros(test_size)
    for pos, i in enumerate(randNum(train_size, train_size+test_size)):
        # print(image_name[i])
        img = mpimage.imread('E:/DESKTOP/Github/DATA/data/RawDataset/' + image_name[i])
        test_data[pos] = torch.tensor(img)
        test_label[pos] = (int(image_name[i].split(',')[2].split('}'[0][0])[0]))-1

    return test_data, test_label


# getTrainData()
# torch.set_printoptions(profile="full")    打印全部tensor