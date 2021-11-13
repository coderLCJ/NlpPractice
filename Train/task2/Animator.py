# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         Animator
# Description:  
# Author:       Laity
# Date:         2021/9/27
# ---------------------------------------------
from IPython import display
from d2l import torch as d2l
from matplotlib import pyplot as plt


class Animator:
    """在动画中绘制数据。"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(9, 7)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        # d2l.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使⽤lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        # d2l.plt.show()
        d2l.plt.show(block=False)
        d2l.plt.pause(0.1)

    def add(self, x, y):
        # 向图表中添加多个数据点
        n = len(y)
        if not hasattr(y, "__len__"):
            y = [y]
            n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
            self.config_axes()
        # display.display(self.fig)
        d2l.plt.show(block=False)
        d2l.plt.pause(0.1)
        display.clear_output(wait=False)

    def stop(self):
        d2l.plt.show(block=True)

    def cla(self):
        d2l.plt.cla()

    def clf(self):
        d2l.plt.clf()

    def close(self):
        d2l.plt.close()

# 用法
# animator = Animator(xlabel='epoch', xlim=[0, 10], ylim=[-5, 5],
#                         legend=['train loss', 'test acc'])
# animator.add(epoch + 1, (loss, acc))