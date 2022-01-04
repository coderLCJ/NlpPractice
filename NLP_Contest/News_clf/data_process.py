# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         data_process
# Description:  
# Author:       Laity
# Date:         2022/1/4
# ---------------------------------------------
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data():
    file = pd.read_csv('E:/DESKTOP/Github/DATA/News_clf/test.csv', sep='\\t', engine='python')
    train_label = torch.LongTensor(file['label'])
    train_text = []
    for text in file['text']:
        t = [0]*58000
        for index, s in enumerate(text.split()):
            t[index] = (int(s))
        train_text.append(t)
    train_text = torch.LongTensor(train_text)
    data_set = TensorDataset(train_text, train_label)
    data = DataLoader(dataset=data_set, batch_size=1, shuffle=True)
    return data


if __name__ == '__main__':
    d = load_data()
    for x, y in d:
        print(x, y)