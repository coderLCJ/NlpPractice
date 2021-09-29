# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/9/29
# ---------------------------------------------
import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt

# img = plt.imread('handwrite/3.jpg')
# print(img)
# plt.imshow(img.reshape(1, 64, 64))
# plt.show()

from PIL import Image

im = Image.open('handwrite/3.jpg')  # 打开图片

pix = im.load()  # 导入像素
width = im.size[0]  # 获取宽度
height = im.size[1]  # 获取长度
print(pix, width, height)
for x in range(width):
    for y in range(height):
        r, g, b = im.getpixel((x, y))
        rgb = (r, g, b)
        # print(rgb)
        im.putpixel((x, y), (0, 1, g, b))
        # if a == 255:
        #     im.putpixel((x, y), (255, 255, 255, 255))


im.save('456.png')