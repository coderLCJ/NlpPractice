# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/10/26
# ---------------------------------------------
import torch
import torch.nn as nn
from torch.autograd import Variable

def t():
    hs = Variable(torch.zeros(2, 2))
    print(hs)


