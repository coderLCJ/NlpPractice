# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         d2l_8_2
# Description:  
# Author:       Laity
# Date:         2021/10/24
# ---------------------------------------------
from d2l import torch
import d2l

def cmp(x):
    return x[1], x[0]


t = lambda x: x + 1
P = {'7': 1, '2': 0, '3': 0, '1': 0}
print(sorted(P.items(), reverse=True, key=cmp))