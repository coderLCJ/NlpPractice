# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         model_test
# Description:  
# Author:       Laity
# Date:         2021/11/5
# ---------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import numpy as np
import demo
import torch.nn
# 读取数据
train = pd.read_csv(r'E:/DESKTOP/Github/DATA/TRAIN_1/test.tsv', sep='\t')
train_origin = pd.read_csv(r'E:/DESKTOP/Github/DATA/TRAIN_1/train.tsv', sep='\t')
# 数据预处理

phraseId = np.array(train['PhraseId'])
size = len(train_origin['Phrase'])
print(len(train_origin['Phrase']))
ct = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
vector = ct.fit_transform(train_origin['Phrase'].append(train['Phrase'])[size:])
word_bag = ct.vocabulary_
one_hot = vector.toarray()
print('vector = ', len(one_hot))
print('words = ', len(word_bag))


train_set = TensorDataset(torch.FloatTensor(one_hot), torch.LongTensor(phraseId))
train_data = DataLoader(train_set, batch_size=128)

print(len(train_data))


net = demo.Net(14324)
state = torch.load('task_1.pt')
net.load_state_dict(state)

print('test')


pre = []
for data in train_data:
    x, id = data
    print(x.shape)
    output = net(x)
    # print(output.shape)
    # print(label.shape)
    pre.append([id, output])
    print(pre)
    p = input()