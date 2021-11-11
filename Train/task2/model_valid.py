# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         model_valid
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
    state = torch.load('../task2/task_1.pt')
    net = NetModel.RNN(bag_size)
    net.load_state_dict(state)

    def valid(size):
        ls = 0.0
        ac = 0.0
        s = 0
        for data in train_data:
            x, label = data
            output = net(x)
            ans = torch.sum(torch.tensor([torch.max(x, -1)[1] for x in output]) == label)
            ac += ans
            s += 1

        print('验证集正确率 = %d/%d' % (ac, s*256))


    start = 153600
    end = 156060
    train_data = loadData.data_loader(start, end)
    valid(len(train_data))
    # exit()



    # 保存模型
    # torch.save(net.state_dict(), 'task_1.pt')
