# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         Lstm_1
# Description:  
# Author:       Laity
# Date:         2021/10/17
# ---------------------------------------------
import torch
from torch import nn
import numpy as np
import matplotlib as mat
mat.use("TkAgg")
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import cv2
torch.manual_seed(1)    # reproducible

LR = 0.02           # learning rate
class LSNN(nn.Module):
    def __init__(self):
        super(LSNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)

        )
        self.hidden = (torch.autograd.Variable(torch.zeros(1, 1, 32)),torch.autograd.Variable(torch.zeros(1, 1, 32)))
        self.out = nn.Linear(32, 1)

    def forward(self,x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out,self.hidden= self.lstm(x,self.hidden)   #hidden_state 也要作为 RNN 的一个输入
        self.hidden=(Variable(self.hidden[0]),Variable(self.hidden[1]))#可以把这一步去掉，在loss.backward（）中加retain_graph=True，主要是Varible有记忆功能，而张量没有
        outs = []    # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):    # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1)

'''
    def forward(self, x, h_n,h_c):
        r_out,  = self.rnn(x, h_state)
        r_out = r_out.view(-1, 32)
        outs = self.out(r_out)
        return outs.view(-1, 10, 1), h_state
'''


lstmNN = LSNN()
optimizer = torch.optim.Adam(lstmNN.parameters(), lr=LR)  # optimize all rnn parameters
loss_func = nn.MSELoss()



  # 要使用初始 hidden state, 可以设成 None

for step in range(100):
    start, end = step * np.pi, (step+2)*np.pi   # time steps
    # sin 预测 cos
    steps = np.linspace(start, end, 10, dtype=np.float32)
    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis,:,np.newaxis]))  # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis,:,np.newaxis]))
    print(x)


    prediction = lstmNN(x)   # rnn 对于每个 step 的 prediction

    loss = loss_func(prediction, y)     # cross entropy loss
    optimizer.zero_grad()               # clear gradients for this training step
    loss.backward()                     # backpropagation, compute gradients
    optimizer.step()
    # apply gradients
    plt.ion()
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw();
    plt.pause(0.05)

    #plt.ioff()
    plt.show()
