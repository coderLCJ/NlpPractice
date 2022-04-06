# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         main
# Description:  
# Author:       Laity
# Date:         2022/3/28
# ---------------------------------------------
import data_loader
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

devices = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    data, vocab_size = data_loader.data_set(100000)
    net = model.Net(vocab_size).to(devices)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    epoch = 128
    i = 0
    acc = 0
    los = 0
    for it in range(epoch):
        print('='*10 + str(it) + '='*10)
        for x, y in tqdm.tqdm(data):
            x = x.to(devices)
            y = y.to(devices)
            predict = net(x)
            loss = criterion(predict, y)
            acc += sum(torch.argmax(predict, -1) == torch.argmax(y, -1))/len(y)
            los += loss

            net.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            if i == 32:
                print('loss = %f  acc = %f' % (los / i, acc / i))
                i = 0
                los = 0
                acc = 0

    torch.save(net.state_dict(), 'net.pt')


if __name__ == '__main__':
    train()

