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

n = nn.Linear(2, 4)
x = torch.tensor([1, 2], dtype=torch.float32)

print(x)
[w, b] = n.parameters()
print(w, b)
print(w.shape)
print(n(x))