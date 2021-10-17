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

file_path = '../../DATA/twitter_en/chat.txt'


p1 = nn.CrossEntropyLoss()
p2 = nn.NLLLoss()

x = torch.tensor([[0., 0., 100.]])
y = torch.tensor([2.])
print(p1(x, y.long()))