# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/10/16
# ---------------------------------------------
import torch
import numpy as np


x = torch.tensor([[1., 2.], [1., 2.]])
net = torch.nn.Linear(2, 1)
n = np.random.randn(10)
print(n)
q = n[np.newaxis, :, np.newaxis, np.newaxis]
print(q)
print(q.shape)