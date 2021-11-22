# 最小二乘法试验
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import pandas as pd
df2 = pd.DataFrame({'c1': [1, 0, 7],
                    'c2': [0, 0, 0],
                    'c3': [3, 0, 9]})
print(df2)