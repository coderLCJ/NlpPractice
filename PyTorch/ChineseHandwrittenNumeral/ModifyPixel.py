# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         ModifyPixel
# Description:  
# Author:       Laity
# Date:         2021/9/29
# ---------------------------------------------
import cv2
import matplotlib.pyplot as plt


image = cv2.imread('t.jpg')
res = cv2.resize(image,(64,64))
cv2.imwrite('test.png', res)  # 写入图片
#cv2.destroyAllWindows()