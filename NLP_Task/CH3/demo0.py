# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         demo0
# Description:  
# Author:       Laity
# Date:         2021/10/22
# ---------------------------------------------
import re

from torch import nn
import torch

text = 'All 432the times have, you ,leave ,my parade.'
print(re.sub('[^A-Za-z.?]+', ' ', text))