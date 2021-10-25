# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         myD2L
# Description:  
# Author:       Laity
# Date:         2021/10/25
# ---------------------------------------------
import re

import torch
from d2l import torch as d2l

def read_time_machine():
    file = open('../data/timemachine.txt', 'r', encoding='utf8')
    text = []
    for line in file.readlines():
        line = re.sub('[^A-Za-z]+', ' ', line).lower()
        text.append(line)
    return text

def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(ch) for ch in lines]
    else:
        return 'error'


if __name__ == '__main__':
    corpus, vocab = d2l.load_corpus_time_machine()
    for i in range(50):
        print(corpus[i], vocab.idx_to_token[corpus[i]])