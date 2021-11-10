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
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.relu(self.fc4(x))
        return x


print('begin')
if __name__ == '__main__':
    # 读取数据
    train = pd.read_csv(r'E:/DESKTOP/Github/DATA/TRAIN_1/train.tsv', sep='\t')
    labels = np.array(train['Sentiment'])
    test = pd.read_csv(r'E:/DESKTOP/Github/DATA/TRAIN_1/test.tsv', sep='\t')
    print(train.shape)
    print(test.shape)
    train_size = train.shape[0]
    test_size = test.shape[0]

    # 数据预处理
    ct = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    vector = ct.fit(pd.concat([train['Phrase'], test['Phrase']]))
    train_vec = ct.transform(train['Phrase'])
    # test_vec = ct.transform(test['Phrase'])
    print(train_vec.shape)
    # print(test_vec.shape)

    # one_hot = vector.toarray()
    # word_bag = ct.vocabulary_

    train_one_hot = train_vec.toarray()
    # test_one_hot = test_vec.toarray()

    print('train_ont size = ', len(train_one_hot))

    input_size = train_vec.shape[1]
    net = Net(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    def valid():
        ls = 0.0
        ac = 0.0
        s = 0
        for data in train_data:
            x, label = data
            output = net(x)
            ans = torch.sum(torch.tensor([torch.max(x, -1)[1] for x in output]) == label)
            # print(ans)
            ac += ans
            s += len(data)
        print('验证集正确率 = ', ac / s)

    for i in range(3):
        for start in range(0, train_size, 12800*2):
            end = start + 12800*2
            if end > train_size:
                end = train_size
                valid()
                break
            print(start, end)

            train_set = TensorDataset(torch.FloatTensor(train_one_hot[start:end]), torch.LongTensor(labels[start:end]))
            train_data = DataLoader(train_set, shuffle=True, batch_size=128)
            print(len(train_set))
            # exit()

            # ==============训练===============
            epoch = 10
            size = len(train_set)
            for e in range(epoch):
                ls = 0.0
                ac = 0.0
                for data in train_data:
                    x, label = data
                    output = net(x)
                    net.zero_grad()
                    # print(output.shape)
                    # print(label.shape)
                    ans = torch.sum(torch.tensor([torch.max(x, -1)[1] for x in output]) == label)
                    # print(ans)
                    ac += ans
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    ls += loss
                print('epoch :', e, ' loss = ', ls / size, 'ac = ', ac / size)
            # ==============训练===============

    # 保存模型
    torch.save(net.state_dict(), 'task_1.pt')
