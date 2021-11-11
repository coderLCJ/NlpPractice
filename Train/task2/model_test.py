# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         model_test
# Description:  
# Author:       Laity
# Date:         2021/11/11
# ---------------------------------------------
import loadData
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
import NetModel
from tqdm import tqdm

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
    bag_size = 17441

    # 数据预处理
    ct = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
    vector = ct.fit(pd.concat([train['Phrase'], test['Phrase']]))
    # train_vec = ct.transform(train['Phrase'])
    test_vec = ct.transform(test['Phrase'])
    # print(train_vec.shape)
    # print(test_vec.shape)

    state = torch.load('../task2/task_1.pt')
    net = NetModel.RNN(bag_size)
    net.load_state_dict(state)

    loader = loadData.test_data_loader()
    pre = []
    for _, x in tqdm(enumerate(loader)):
        y = net(torch.LongTensor(x))
        ans = torch.max(y, -1)[1]
        pre.append(int(ans))

    print(len(pre))
    loadData.test_data['Sentiment'] = pre
    loadData.test_data[['PhraseId', 'Sentiment']].set_index('PhraseId').to_csv('test_pre2.csv')
