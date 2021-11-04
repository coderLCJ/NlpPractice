# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         demo_0
# Description:  
# Author:       Laity
# Date:         2021/11/4
# ---------------------------------------------
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import TensorDataset, DataLoader

ar1 = '今天 今天 天气 不错 我们 愉快 玩耍'
ar2 = '今天 锻炼 舒服 天气 一般'
ar3 = '天气 糟糕 糟糕'
text = [ar1, ar2, ar3]
ct = CountVectorizer()
print(ct.fit_transform(text))
print(ct.fit_transform(text).todense())
print(ct.vocabulary_)

train_x = [1, 2, 3, 4]
label = [0, 1, 0, 1]
train = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(label))
data = DataLoader(train, shuffle=True, batch_size=2)

