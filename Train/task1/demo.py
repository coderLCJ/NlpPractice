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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

print('begin')
if __name__ == '__main__':
    # 读取数据
    train = pd.read_csv(r'E:/DESKTOP/Github/DATA/TRAIN_1/train.tsv', sep='\t')
    labels = np.array(train['Sentiment'])

    # 数据预处理
    ct = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
    vector = ct.fit_transform(train['Phrase'])
    one_hot = vector.toarray()
    word_bag = ct.vocabulary_
    print('vector = ', len(one_hot))
    print('words = ', len(word_bag))

    train_set = TensorDataset(torch.FloatTensor(one_hot), torch.LongTensor(labels))
    train_data = DataLoader(train_set, shuffle=True, batch_size=128)

# 定义网络

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

def train():
    epoch = 30
    input_size = len(word_bag)
    net = Net(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.2)
    size = len(train_data)
    for e in range(epoch):
        ls = 0.0
        ac = 0.0
        for data in train_data:
            x, label = data
            output = net(x)
            net.zero_grad()
            # print(output.shape)
            # print(label.shape)
            ans = torch.sum(torch.tensor([max(x) for x in output]) == label)
            # print(ans)
            ac += ans
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            ls += loss
        print('epoch :', e, ' loss = ', ls / size, 'ac = ', ac / size / 128)


train()
# torch.save(net.state_dict(), 'task_1.pt')