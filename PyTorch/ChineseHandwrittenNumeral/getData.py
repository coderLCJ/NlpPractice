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

train_size = 4
test_size = 1310
image_name = os.listdir('../data/RawDataset')

# print(image_data[0], len(image_data))
# for i in range(10):
#     print(image_data[i])
#     print(image_data[i].split(',')[2].split('}'[0]))

def getTrainData():
    train_data = torch.zeros((1, train_size), dtype=torch.float32)
    train_label = torch.zeros((1, train_size), dtype=torch.float32)
    for i in range(train_size):
        img = mpimage.imread('../data/RawDataset/' + image_name[i])
        # train_data.
        # train_label.append(image_name[i].split(',')[2].split('}'[0][0]))

    print(type(train_data))
    print(train_label)


getTrainData()