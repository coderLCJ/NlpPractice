# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test'
# Description:  
# Author:       Laity
# Date:         2021/10/15
# ---------------------------------------------
import torch
import torch.nn.functional as F


x = torch.Tensor([1, 23, 54, 65])
print(F.sigmoid(x))