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
test_vec = ct.transform(test['Phrase'])
test_one_hot = test_vec.toarray()

input_size = 16790
net = model_train.Net(input_size)
state = torch.load('task_1_.pt')
net.load_state_dict(state)

print('test')

pre = []
for inputs in test_one_hot:
    out = net(torch.Tensor(inputs))
    ans = torch.max(out, -1)[1]
    pre.append(int(ans))

test['Sentiment'] = pre
test[['PhraseId', 'Sentiment']].set_index('PhraseId').to_csv('test_pre.csv')