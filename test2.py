# coding = utf8
# 最小二乘法试验
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 样本数据
Xi = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
Yi = np.array([0, 1.27, 2.16, 2.86, 3.44, 3.87, 4.15, 4.37, 4.51, 4.58, 4.62, 4.64])

# 拟合函数
def func(p, x):
    k, b = p
    return k * np.exp(b / x)

# 误差error
def error(p, x, y):
    return func(p, x) - y


p0 = np.array([1, -1])  # 参数初始化
Para = leastsq(error, p0, args=(Xi[1:], Yi[1:]))
k, b = Para[0]
plt.scatter(Xi, Yi, c="red", label="样本点", linewidth=1)  # 样本点
x = np.linspace(1e-6, 55, 100)
# k = 5.2151048
# b = -7.4961942
y = k * np.exp(b / x)
plt.plot(x, y, c="blue", label="拟合曲线", linewidth=1)  # 拟合直线
plt.legend()
plt.show()