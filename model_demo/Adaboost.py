# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         Adaboost
# Description:  
# Author:       Laity
# Date:         2022/1/21
# ---------------------------------------------
from time import sleep

import numpy as np
from math import log, e, exp


class Adaboost:
    def __init__(self, N):
        self.D = np.array([1 / N] * N)
        self.cls = []
        self.a = []
        self.N = N

    # 计算误差率
    def error(self, x, y, Gx, w):
        e = 0
        for i in range(len(x)):
            #print(i, Gx(x[i]), y[i])
            if Gx(x[i]) != y[i]:
                e += w[i]
            #print(e)
        return float('%.3f' % e)

    # 信号函数
    def sign(self, x):
        return np.array([1 if i > 0 else -1 for i in x])

    # 组合成强分类器
    def f(self, x):
        all_s = np.zeros(self.N)
        for cls_i in self.cls:
            s = np.zeros(self.N)
            for i in range(self.N):
                s[i] = cls_i(x[i])
            all_s += s
        return self.sign(all_s)

    # 选择最优划分
    def choose_cls(self):
        ret_G_x = None
        ret_v = 0
        ei = self.N
        for i in range(self.N):
            v = i + 0.5
            G_x = lambda x_t: 1 if x_t < v else -1
            # print('v = ', v, 'e = ', self.error(x, y, G_x, self.D))
            if self.error(x, y, G_x, self.D) < ei:
                ret_G_x = G_x
                ret_v = v
                ei = self.error(x, y, G_x, self.D)
        return ret_G_x, ret_v, ei

    # 训练多个弱分类器
    def fit(self, x, y):
        while True:
            if len(self.cls) != 0 and sum(self.f(x) == y) == 0:
                break
            Gx_i, v, ei = self.choose_cls()
            print(v)
            ai = log((1 - ei) / (ei + 0.000000001), e) / 2
            Gx = lambda t_x: ai if t_x < v else -ai
            self.a.append(ai)
            Zm = sum([self.D[i] * exp(-ai * y[i] * Gx_i(x[i])) for i in range(len(x))])
            new_D = [self.D[i] / Zm * exp(-ai * y[i] * Gx_i(x[i])) for i in range(len(x))]
            self.D = new_D
            self.cls.append(Gx)


