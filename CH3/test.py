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


net = nn.Embedding(6, 6)
id = torch.tensor([1, 2, 3, 4, 5, 6])
print(net(id[0]))
print(net(torch.tensor(1)))
