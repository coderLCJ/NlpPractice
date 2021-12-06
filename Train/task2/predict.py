# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         predict
# Description:  
# Author:       Laity
# Date:         2021/12/6
# ---------------------------------------------
import pandas as pd

test_data = pd.read_csv('../../../DATA/TRAIN_1/test.tsv', sep='\t')
output = open('OUTPUT.txt', 'r', encoding='utf8')

predict = []
for line in output.readlines():
    predict.append(line.split()[0][1])

test_data['Sentiment'] = predict
test_data[['PhraseId', 'Sentiment']].set_index('PhraseId').to_csv('Bert_pre.csv')