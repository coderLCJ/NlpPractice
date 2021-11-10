# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/11/10
# ---------------------------------------------
import torch
import torch.nn as nn

em = nn.Embedding(10, 6)
i = torch.ones(6, 28, dtype=torch.int32)
print(em(i).shape)
