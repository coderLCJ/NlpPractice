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


out = torch.ones(128)
pre = torch.zeros(128, 5)
pre[0] = torch.tensor([0., 0., 2., 0., 0.])
print(torch.sum(torch.tensor((torch.tensor([max(x) for x in pre]) == out))))