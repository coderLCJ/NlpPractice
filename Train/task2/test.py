# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/11/10
# ---------------------------------------------
from time import sleep

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

x = torch.randn(128, 28, 256)
x = x.permute(0, 2, 1)
print(x.shape)
net = nn.Conv1d(256, 100, kernel_size=(2,))
y = net(x)
print(y.shape)