# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test2
# Description:  
# Author:       Laity
# Date:         2021/9/29
# ---------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from PIL import Image

t = plt.imread('test.png')
t = torch.tensor(t)
t = t.sum(dim=2)
print(t.size())
plt.imshow(t)
plt.show()

