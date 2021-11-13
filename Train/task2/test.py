# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/11/10
# ---------------------------------------------
from time import sleep

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Animator import *

animator = Animator(xlabel='epoch', xlim=[0, 10], ylim=[-5, 5],
                        legend=['train loss', 'test acc'])
animator.add(1, (1, 2))
animator.add(3, (2, 3))
sleep(3)
animator.clf()
animator.add(2, (1, 2))
animator.stop()