import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from math import log

Xi = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
Yi = np.array([0, 1.27, 2.16, 2.86, 3.44, 3.87, 4.15, 4.37, 4.51, 4.58, 4.62, 4.64])

s = 0
for i in range(1, 12):
    s += log(Yi[i])/Xi[i]

A = (13.639649 - 0.530331 * (0.603975 / 0.062321)) / (11 - 0.603975 * (0.603975 / 0.062321))
# b = (13.639649 - 0.530331 * (11 / 0.603975)) / (0.603975 - 0.602321 * (11 / 0.603975))
# print(A)
# print(b)
b = (13.639649 - 11 * A) / 0.603975
print(b)
print(A * 11 + 0.603975 * b)
print(math.exp(A))