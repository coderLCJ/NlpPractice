# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         model_test
# Description:  
# Author:       Laity
# Date:         2021/11/5
# ---------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import numpy as np
import model_train
import torch.nn as nn
# 读取数据
test = pd.read_csv(r'E:/DESKTOP/Github/DATA/TRAIN_1/test.tsv', sep='\t')
train = pd.read_csv(r'E:/DESKTOP/Github/DATA/TRAIN_1/train.tsv', sep='\t')
# 数据预处理

ct = TfidfVectorizer(stop_words='english')
vector = ct.fit(pd.concat([train['Phrase'], test['Phrase']]))
test_vec = ct.transform(test['Phrase'])
test_one_hot = test_vec.toarray()

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
        x = self.fc4(x)
        return x


input_size = 17441
net = Net(input_size)
state = torch.load('../task2/task_1.pt')
net.load_state_dict(state)

print('test')

pre = []
for inputs in test_one_hot:
    out = net(torch.Tensor(inputs))
    ans = torch.max(out, -1)[1]
    pre.append(int(ans))

test['Sentiment'] = pre
test[['PhraseId', 'Sentiment']].set_index('PhraseId').to_csv('test_pre2.csv')