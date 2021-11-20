# 问题2：三次插值实验
from sympy import symbols, expand
import numpy as np
import matplotlib.pyplot as plt

# 五个时刻的数据
ti = [0, 5, 10, 15, 20]
p = [0, 1.27, 2.16, 2.86, 3.44]

M = [0, -0.034685, 0.04754, 0.04115, 0]
h = [5, 5, 5, 5]
S = []

# 求出公式
t = symbols('t')
for j in range(0, 4):
    f = M[j] * (ti[j + 1] - t) ** 3 / (6 * h[j]) + M[j + 1] * (t - ti[j]) ** 3 / (6 * h[j]) + (
                p[j] - (M[j] * h[j] ** 2) / 6) * ((ti[j + 1] - t) / h[j]) + (p[j + 1] - M[j + 1] * h[j] ** 2 / 6) * (
                    t - ti[j]) / h[j]
    S.append(expand(f))
    print('S(t) = %s  t=>[%d, %d]' % (S[-1], ti[j], ti[j+1]))

# 绘出图像
plt.scatter(ti, p, c="red", label='Sample point', linewidth=1)  # 样本点
for i in range(len(ti)-1):
    t = np.linspace(ti[i], ti[i+1], 1000)
    p = eval(str(S[i]))
    plt.plot(t, p, c="blue", label="Fitting curve"if i == 0 else None,linewidth=1)  # 拟合直线
plt.legend()
plt.show()
