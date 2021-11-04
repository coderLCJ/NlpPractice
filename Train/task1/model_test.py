# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test_model
# Description:  
# Author:       Laity
# Date:         2021/11/4
# ---------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import numpy as np
import demo
import torch.nn
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

net = demo.Net(len(word_bag))
state = torch.load('task_1.pt')
net.load_state_dict(state)

print('test')
# for i in train_data:
#     x, y = i
#     out = net(x)
#     print(out)
#     print(y)
#     ac = torch.sum(torch.tensor([max(x) for x in out]) == y)
#     print(ac)
#     p = input()
loss = torch.nn.CrossEntropyLoss()
for data in train_data:
    x, label = data
    output = net(x)
    # print(output.shape)
    # print(label.shape)
    ac = torch.sum(torch.tensor([max(x) for x in output]) == label)
    l = loss(output, label)
    print('ac = ', ac, '/128', ' loss = ', l)