# 最小二乘法试验
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 样本数据
Xi = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
Yi = np.array([0, 1.27, 2.16, 2.86, 3.44, 3.87, 4.15, 4.37, 4.51, 4.58, 4.62, 4.64])

# 拟合函数
def func(p, x):
    a, b = p
    return a * np.exp(b / x)

# 误差error
def error(p, x, y):
    return func(p, x) - y

p0 = np.array([1, -1])  # 参数初始化
Para = leastsq(error, p0, args=(Xi[0:5], Yi[0:5]))    # 拟合曲线
a, b = Para[0]
plt.scatter(Xi[0:5], Yi[0:5], c="red", label="Sample point", linewidth=1)  # 样本点
x = np.linspace(1e-6, 20, 100)
y = a * np.exp(b / x)
plt.plot(x, y, c="blue", label="Fitting curve", linewidth=1)  # 拟合直线
plt.legend()
plt.show()