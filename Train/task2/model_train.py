# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         demo
# Description:
# Author:       Laity
# Date:         2021/11/4
# ---------------------------------------------
import pandas as pd
import unicodedata, re, string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader
import NetModel
import loadData

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def valid(s, e, net):
    data_loader = loadData.data_loader(s, e)
    ac = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        p = net(x)
        ac += sum((torch.max(p.cpu(), -1)[1] == y.cpu()).numpy())
    print('ac: %d' % ac)

def train():
    bag_size = len(loadData.words_bag)
    train_data_size = loadData.train_data_size
    net = NetModel.CNN(bag_size).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(params=net.parameters())


    for i in range(10):
        sep = 12800
        for begin in range(0, train_data_size, sep):
            if begin + sep > train_data_size:
                sep = train_data_size - begin
                torch.save(net.state_dict(), 'task_1.pt')
                valid(begin, begin + sep, net)
                break
            print(begin, begin + sep)

            data_loader = loadData.data_loader(begin, begin+sep)
            t = 0
            epoch = 160
            for e in range(epoch):
                ac = 0
                ls = 0
                size = len(data_loader) * data_loader.batch_size
                # print('data loader size = ', size)
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    net.zero_grad()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    ls += loss
                    ac += sum((torch.max(outputs.cpu(), -1)[1] == labels.cpu()).numpy())
                    # print(ac)
                t += 1
                print('epoch: %d  loss: %f  ac: %f' % (e, ls/size*data_loader.batch_size, ac/size))
        torch.save(net.state_dict(), 'task_2.pt')


train()