# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2022/3/22
# ---------------------------------------------
import numpy
import numpy as np

import torch
import tensorflow.keras as kr

data_id = [[1, 2, 3], [2, 3, 4, 4, 6, 6]]
max_length = 10
x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
print(x_pad)

y_pad = [[0]*(max(0, max_length-len(i)))+i for i in data_id]
print(np.array(y_pad))