# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         demo0
# Description:  
# Author:       Laity
# Date:         2021/9/25
# ---------------------------------------------
import torch
import torch.nn as nn

rnn = nn.RNN(5, 6, 1)
input = torch.randn(1, 3, 5)
h0 = torch.randn(1, 3, 6)
print(h0)
output, hn = rnn(input, h0)

print(output)
