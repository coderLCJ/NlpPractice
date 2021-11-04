# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/11/2
# ---------------------------------------------

import torch.nn.functional as F
import torch
import torch.nn


input = torch.randn(2, 2, 2)
print(input)
input.resize(2, 4)
print(input)