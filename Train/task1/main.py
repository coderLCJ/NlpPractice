# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         main
# Description:  
# Author:       Laity
# Date:         2021/11/4
# ---------------------------------------------
import pandas
import torch

def password(w):
    sbol = ',.:?!'
    return w not in sbol and w.isalpha()

def readData(name):
    words2id = {}
    id2words = {}
    num = 0
    size = 0
    df = pandas.read_csv('E:/DESKTOP/Github/DATA/TRAIN_1/%s.tsv' % name, sep='\t')
    for s in df.itertuples():
        word_ = s[3].split()
        for word in word_:
            word = word.lower()
            if word not in words2id and password(word):
                words2id[word] = num
                id2words[num] = word
                num += 1
        size += 1
    # words = 14324
    batch_size = 64
    words_size = len(words2id)
    # print(len(words2id))
    # for i in range(40):
    #     print(id2words[i])
    one_hot = []
    label = []
    t = 0
    for s in df.itertuples():
        code = [0] * words_size
        for word in s[3].split():
            word = word.lower()
            if password(word):
                code[words2id[word]] = 1
        if name == 'train':
            label.append(s[4])
        one_hot.append(code)
        t += 1
        if t % 1000 == 0:
            print(t)
    return one_hot, label, size

# readData('train')

def getData():
    data, label, size = readData('train')
    train_data, train_label = data[:size*0.7], label[:size*0.7]
    valid_data, valid_label = data[size*0.7:], label[size*0.7:]
    print(train_data[0][:50])


getData()

