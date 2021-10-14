# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/10/10
# ---------------------------------------------
import torch

net = torch.nn.Linear(4, 2)
[W, b] = net.parameters()
print(W)
print(b)