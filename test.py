# -*- coding: utf-8 -*-
from pandas import Series,DataFrame
import numpy as np
import  pandas as pd


df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df3 = df1.add(df2, fill_value=4)


print(df3)