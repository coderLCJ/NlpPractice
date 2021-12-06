# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         process_data
# Description:  
# Author:       Laity
# Date:         2021/12/6
# ---------------------------------------------
import pandas as pd


train_data = pd.read_csv('../../../DATA/TRAIN_1/train.tsv', sep='\t')
label = open('data/label', 'w', encoding='utf8')
seq_in = open('data/seq.in', 'w', encoding='utf8')
seq_out = open('data/seq.out', 'w', encoding='utf8')

test_label = open('dev/label', 'w', encoding='utf8')
test_seq_in = open('dev/seq.in', 'w', encoding='utf8')
test_seq_out = open('dev/seq.out', 'w', encoding='utf8')

test_data = pd.read_csv('../../../DATA/TRAIN_1/test.tsv', sep='\t')
test_save = open('test.txt', 'w', encoding='utf8')

def make_test():
    for line in test_data['Phrase']:
        test_save.write(line + '\n')

make_test()

def make_train():
    for index, line in enumerate(train_data['Phrase']):
        # print(line, train_data['Sentiment'][1])
        words_len = len(list(line.split()))
        BIO = ('O '*words_len).strip()
        if index > 14000:
            test_seq_in.write(line + '\n')
            test_seq_out.write(BIO + '\n')
            test_label.write(str(train_data['Sentiment'][index]) + '\n')
            continue
        seq_in.write(line+'\n')
        seq_out.write(BIO+'\n')
        label.write(str(train_data['Sentiment'][index])+'\n')
        # print(BIO)