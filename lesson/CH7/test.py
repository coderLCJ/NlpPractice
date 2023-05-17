# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         RNN to LSTM
# Description:
# Author:       Laity
# Date:         2021/10/16
# ---------------------------------------------
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 10
INPUT_SIZE = 1

# plt.plot(steps, x_np, 'c-', label='input(sin)')
# plt.plot(steps, y_np, 'r-', label='target(cos)')
# plt.legend(loc='best')
# plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch ,hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        r_out = r_out.reshape(-1,32)
        outs = self.out(r_out)
        return outs, h_state


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters())
criterion = nn.MSELoss()
EPOCHS = 300
h_state = None

for step in range(EPOCHS):
    start, end = step*np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32) # 均匀分成 TIME_STEP 份
    # np.newaxis: 插入新维度
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[:, np.newaxis])
    outs, h_state = rnn(x, h_state)

    h_state = h_state.detach()  # 这一步很重要 不计算梯度将隐藏层数值传递
    rnn.zero_grad()
    loss = criterion(outs, y)
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        # a.flatten()：a是个数组，a.flatten()
        # 就是把a降到一维，默认是按行的方向降 。
        # a.flatten().A：a是个矩阵，降维后还是个矩阵，矩阵.A（等效于矩阵.getA()）变成了数组
        plt.plot(steps, y_np.flatten(), 'r-', label='cos')
        plt.plot(steps, outs.data.numpy().flatten(), 'b-', label='predict')
        plt.legend(loc='best')
        plt.draw()
        plt.pause(1)
        plt.clf()
        # clf()  # 清图
        # cla()  # 清坐标轴
        # close()  # 关窗口