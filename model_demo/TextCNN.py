# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         TextCNN
# Description:  
# Author:       Laity
# Date:         2021/12/29
# ---------------------------------------------
import torch
import torch.nn as nn

textCNN = nn.Conv1d(in_channels=256, out_channels=100, kernel_size=(2,))
x = torch.randn(128, 32, 256)   # 128批量，32个单词，256维度词向量
x = x.permute(0, 2, 1)  # 在最后一维做卷积（单词间做卷积），交换最后两个维度
y = textCNN(x)
print(y.shape)