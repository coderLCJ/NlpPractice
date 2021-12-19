# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         loadData
# Description:
# Author:       Laity
# Date:         2021/11/10
# ---------------------------------------------
import torch.nn as nn
import torch

input = torch.randn(32, 64, 128)
LSTM = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=False)
output, (hn, cn) = LSTM(input)
print(output.shape)
print(hn.shape)
print(cn.shape)