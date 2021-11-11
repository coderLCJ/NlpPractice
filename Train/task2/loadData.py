# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         loadData
# Description:  
# Author:       Laity
# Date:         2021/11/10
# ---------------------------------------------
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


train_data = pd.read_csv('E:/DESKTOP/Github/DATA/TRAIN_1/train.tsv', sep='\t')
test_data = pd.read_csv('E:/DESKTOP/Github/DATA/TRAIN_1/test.tsv', sep='\t')

ct = CountVectorizer(stop_words='english')
vector = ct.fit(pd.concat([train_data['Phrase'], test_data['Phrase']]))
words_bag = ct.vocabulary_


train_data_size = train_data['Phrase'].shape[0]
test_data_size = test_data['Phrase'].shape[0]
print(train_data_size, test_data_size)


sentence_size = 28

def data_loader(begin, end):
    ret_vec = []
    for corp in train_data['Phrase'][begin:end]:
        # print(corp)
        input_vector = np.zeros(sentence_size)
        index = 0
        for word in corp.split():
            if word.lower() in words_bag:
                input_vector[index] = words_bag[word.lower()]
                index += 1
            if index >= 28:
                break
        # print(input_vector)
        ret_vec.append(input_vector)
    labels = train_data['Sentiment'][begin:end]

    ret_set = TensorDataset(torch.LongTensor(np.array(ret_vec)), torch.LongTensor(np.array(labels)))
    ret_loader = DataLoader(ret_set, batch_size=256, shuffle=True)
    return ret_loader

def test_data_loader():
    ret_vec = []
    for corp in test_data['Phrase']:
        # print(corp)
        input_vector = np.zeros(sentence_size)
        index = 0
        for word in corp.split():
            if word.lower() in words_bag:
                input_vector[index] = words_bag[word.lower()]
                index += 1
            if index >= 28:
                break
        # print(input_vector)
        ret_vec.append(input_vector)
    return np.array(ret_vec)