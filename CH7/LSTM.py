# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         LSTM
# Description:  
# Author:       Laity
# Date:         2021/10/17
# ---------------------------------------------
# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         RNN
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

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, t):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch ,hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.lstm(x, t)
        # exit(0)
        r_out = r_out.reshape(-1,32)
        outs = self.out(r_out)
        return outs, h_state


rnn = LSTM()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)
criterion = nn.MSELoss()
EPOCHS = 3000
h = torch.zeros(1, 1, 32)
c = torch.zeros(1, 1, 32)

for step in range(EPOCHS):
    start, end = step*np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32) # 均匀分成 TIME_STEP 份
    # np.newaxis: 插入新维度
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[:, np.newaxis])
    outs, t = rnn(x, (h, c))
    h = t[0]
    c = t[1]
    h = h.detach()
    c = c.detach()

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