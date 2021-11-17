# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/11/15
# ---------------------------------------------
# task1
import torch

x = torch.randn(28, 256)
print(x)
x = x.unsqueeze(1)
print(x.shape)
