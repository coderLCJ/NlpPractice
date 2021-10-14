# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         process
# Description:  
# Author:       Laity
# Date:         2021/10/10
# ---------------------------------------------

def splitWord(text):
    text = text.replace('.', ' .')
    text = text.lower()
    words = text.split(' ')
    return words

def pre_process(text):
    words = splitWord(text)
    word2id = {}
    id2word = {}
    for word in words:
        if word not in word2id:
            new_id = len(word2id)
            word2id[word] = new_id
            id2word[new_id] = word
    corpus = [word2id[w] for w in words]
    return word2id, id2word, corpus


pre_process('you say goodbye and i say hello.')