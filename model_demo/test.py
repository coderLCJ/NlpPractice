# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2022/1/25
# ---------------------------------------------
import torch
import torch.nn.functional as F

x = torch.rand(10, 4)
print(x)
m = torch.nn.BatchNorm1d(4)
print(m(x))
