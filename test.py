# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/9/25
# ---------------------------------------------
import os

import torch

t = torch.randn(2, 1, 3)
print(t)

t = torch.randn(2, 1, 3).squeeze(1)
print(t)

