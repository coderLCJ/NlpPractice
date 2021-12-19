# def f(x):
#     return (3 * x + 1) ** (1 / 3) # 定义迭代函数
#
# def iteration():
#     x_k = 2
#     for k in range(6):  # 迭代5次
#         print(x_k)
#         x_k = f(x_k)
#     return x_k


# iteration()


# def f(x):
#         return x ** 3 - 3 * x - 1  # 定义函数
#
# def iteration():
#     x0, x1 = 1, 2
#     x_k = 0
#     for t in range(6):  # 迭代5次
#         x_k = (x0 + x1) / 2 # 取中点
#         f_k = f(x_k)    # 计算函数值
#         print(x0, x1, x_k, f_k)
#         if f_k == 0:    # 找到根 返回
#             return x_k
#         elif f_k < 0:   # 用中点替代x0
#             x0 = x_k
#         else:
#             x1 = x_k    # 用中点替代x1
#     return x_k
# iteration()

def f(x):
    return (2 * x**3 + 1) / (3 * (x - 1) * (x + 1))


x = 1.2
x = f(x)
print(x)
x = f(x)
print(x)
print(x - 1.87938524 < 0.5 * 10**-3)
x = f(x)
print(x)
print(x - 1.87938524 < 0.5 * 10**-3)
x = f(x)
print(x)
print(x - 1.87938524 < 0.5 * 10**-3)
x = f(x)
print(x)
print(x - 1.87938524 < 0.5 * 10**-3)
x = f(x)
print(x)
print(x - 1.87938524 < 0.5 * 10**-3)