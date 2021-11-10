# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         run_test
# Description:  
# Author:       Laity
# Date:         2021/11/9
# ---------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import numpy as np
import model_train
import torch.nn
# 读取数据
test = pd.read_csv(r'E:/DESKTOP/Github/DATA/TRAIN_1/test.tsv', sep='\t')
train = pd.read_csv(r'E:/DESKTOP/Github/DATA/TRAIN_1/train.tsv', sep='\t')
# 数据预处理

ct = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
vector = ct.fit(pd.concat([train['Phrase'], test['Phrase']]))


input_size = 16790
net = model_train.Net(input_size)
state = torch.load('task_1_.pt')
net.load_state_dict(state)

print('test')

pre = []
for i in range(100):
    inputs = input()
    test_vec = ct.transform([inputs])
    test_one_hot = test_vec.toarray()
    out = net(torch.Tensor(test_one_hot))
    ans = torch.max(out, -1)[1]
    if ans == 0:
        print('悲观')
    elif ans == 1:
        print('有点悲观')
    elif ans == 2:
        print('中性')
    elif ans == 3:
        print('有点乐观')
    elif ans == 4:
        print('乐观')
