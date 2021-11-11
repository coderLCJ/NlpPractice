# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/11/10
# ---------------------------------------------
import torch
import torch.nn as nn

outputs = torch.randn(10, 3)
lab = torch.zeros(10)
print(outputs)
print(sum((torch.max(outputs, -1)[1] == lab).numpy()))