# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         TowLayerNet
# Description:  
# Author:       Laity
# Date:         2021/10/10
# ---------------------------------------------
from Trainer import *

class Net:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        # 初始化权重
        W1 = np.random.randn(I, H)
        b1 = np.random.rand(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)
        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        # 整理权重
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(1, 2)
net = Net(2, 4, 3)
y = net.predict(x)
print(y)
