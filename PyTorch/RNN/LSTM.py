# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         LSTM
# Description:  
# Author:       Laity
# Date:         2021/9/25
# ---------------------------------------------
import torch.nn as nn
import torch

rnn = nn.LSTM(5, 6, 2)
input = torch.randn(1, 3, 5)
h0 = torch.randn(2, 3, 6)
c0 = torch.randn(2, 3, 6)
output, (hn, cn) = rnn(input)
print(output.shape)
print(hn.shape)
print(cn.shape)