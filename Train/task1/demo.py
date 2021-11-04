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

# 导入数据
# %%
train = pd.read_csv(r'E:/DESKTOP/Github/DATA/TRAIN_1/train.tsv', sep='\t')
test = pd.read_csv(r'E:/DESKTOP/Github/DATA/TRAIN_1/test.tsv', sep='\t', header=0, index_col=0)
labels = np.array(train['Sentiment'])  # 原本：torch.Tensor( train['Sentiment'])

# First step - tokenizing phrases
Vectorizer = CountVectorizer(max_df=0.95, min_df=5,stop_words='english')#去除停用词效果确实好了一点点
# (a,b),（ 单词所在得句子，单词所在词袋中得位置）出现的次数
train_CountVectorizer = Vectorizer.fit_transform(train['Phrase'])
train_bag = Vectorizer.vocabulary_
train_one_hot = train_CountVectorizer.toarray()#纤细模型one-hot,但是RNN/CNN都是索引位置
print(len(train_bag))
print(len(train_one_hot))
print(train_one_hot[0][:50])
print(len(train_one_hot[0]))