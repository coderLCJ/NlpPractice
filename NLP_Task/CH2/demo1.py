# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         demo1
# Description:  
# Author:       Laity
# Date:         2021/10/22
# ---------------------------------------------
from text import kite_origin
import operator

words = {}
del_char = ',.?:;!\t\n'
words_num = 0
for word in kite_origin.split():
    for i in del_char:
        if i in word:
            word = word.replace(i, '')
    word = word.lower()
    words_num += 1
    if word not in words:
        words[word] = 1
    else:
        words[word] += 1

print(sorted(words.items(), key=operator.itemgetter(1), reverse=True))

tf = {}
idf = {}
tfidf = {}
for k, v in words.items():
    tf[k] = v / words_num

print(sorted(tf.items(), key=operator.itemgetter(1),reverse=True))