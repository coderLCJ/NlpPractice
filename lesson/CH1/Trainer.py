# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         Trainer
# Description:  
# Author:       Laity
# Date:         2021/10/10
# ---------------------------------------------
import numpy as np

class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out

