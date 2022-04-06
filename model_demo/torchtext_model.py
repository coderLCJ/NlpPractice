# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         torchtext
# Description:  
# Author:       Laity
# Date:         2022/2/8
# ---------------------------------------------
import pandas as pd
from torchtext.legacy import data
import torch
# file = pd.read_csv('E:/DESKTOP/Github/DATA/TRAIN_1/train.tsv', sep='\t')
# file.iloc[:100].to_csv('sample.csv', index=False)

BATCH_SIZE = 16
DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

# file = pd.read_csv('sample.csv')
pd.set_option('display.max_columns', 10)


PAD_TOKEN = '<pad>'
TEXT = data.Field(sequential=True,batch_first=True, lower=True, pad_token=PAD_TOKEN)
LABEL = data.Field(sequential=False, batch_first=True, unk_token=None)

#读取数据
datafields = [("PhraseId", None), # 不需要的filed设置为None
              ("SentenceId", None),
              ('Phrase', TEXT),
              ('Sentiment', LABEL)]
train_data = data.TabularDataset(path='sample.csv', format='csv', fields=datafields)

#构建词典，字符映射到embedding
#TEXT.vocab.vectors 就是词向量
TEXT.build_vocab(train_data,  vectors= 'glove.6B.50d',   #可以提前下载好
                 unk_init= lambda x:torch.nn.init.uniform_(x, a=-0.25, b=0.25))
LABEL.build_vocab(train_data)


#得到索引，PAD_TOKEN='<pad>'
PAD_INDEX = TEXT.vocab.stoi[PAD_TOKEN]
TEXT.vocab.vectors[PAD_INDEX] = 0.0
#构建迭代器
train_iterator = data.BucketIterator(train_data, batch_size=BATCH_SIZE, train=True, shuffle=True,device=DEVICE)

