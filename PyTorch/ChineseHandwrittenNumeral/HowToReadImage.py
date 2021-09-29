# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         HowToReadImage
# Description:  
# Author:       Laity
# Date:         2021/9/29
# ---------------------------------------------
# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         main
# Description:
# Author:       Laity
# Date:         2021/9/29
# ---------------------------------------------
import io
import torch
import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from skimage import io,transform
import pandas as pd


#设置显示的最大列、宽等参数，消除打印不完全中间的省略号
pd.set_option("display.width",1000)
#加了这一行那表格就不会分段出现了
pd.set_option("display.width",1000)
#显示所有列
pd.set_option("display.max_columns",None)
#显示所有行
pd.set_option("display.max_rows",None)


lena = mpimg.imread('../data/RawDataset/Locate{1,1,1}.jpg') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
print(lena.shape) #(512, 512, 3)
print(type(lena))
for i in range(64):
    for j in range(64):
        print(lena[i][j], end=' ')
    print('')
plt.imshow(lena) # 显示图片
plt.axis() # 不显示坐标轴
plt.show()