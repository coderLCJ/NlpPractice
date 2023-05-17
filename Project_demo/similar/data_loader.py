# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         data_loader
# Description:  
# Author:       Laity
# Date:         2022/3/28
# ---------------------------------------------
import pandas as pd
import torch
import numpy as np

class Vocab:
    def __init__(self, sentences1, sentences2):
        sentences = sentences1 + [' '] + sentences2
        self.vocab_size = 2
        self.word2id = {'<PAD>': 0, '<UNK>': 1}
        self.id2word = {0: '<PAD>', 1: '<UNK>'}
        self.label_dict = {'contradiction':0, 'neutral': 1, 'entailment': 2, '<UNK>': 3}
        self.label_id = {0: 'contradiction', 1: 'neutral', 2: 'entailment', 3: '<UNK>'}
        self.stop_sign = '.,/?;:!'
        for sentence in sentences:
            # 去标点符号
            for sign in self.stop_sign:
                sentence = sentence.replace(sign, ' ')
            for word in sentence.split():
                if word not in self.word2id:
                    word = word.lower()
                    self.word2id[word] = self.vocab_size
                    self.id2word[self.vocab_size] = word
                    self.vocab_size += 1

    def __getitem__(self, item):
        if item in self.word2id:
            return self.word2id[item]
        else:
            return self.id2word[item]

    def label2id(self, label_text):
        # label制作成one_hot
        label = [0]*4
        if label_text not in self.label_dict:
            label_text = '<UNK>'
        label[self.label_dict[label_text]] = 1
        return label

    def id2label(self, label_num):
        if label_num not in self.label_id:
            return '<UNK>'
        return self.label_id[label_num]

    def sentence2id(self, sentence):
        sentence_id = []
        for sign in self.stop_sign:
            sentence = sentence.replace(sign, ' ')
        for word in sentence.split():
            word = word.lower()
            sentence_id.append(self.word2id[word])
        return sentence_id

    def padding(self, sentence_id, sentence_size):
        for i in range(len(sentence_id), sentence_size):
            sentence_id.append(self.word2id['<PAD>'])
        # 截取超出长度
        return sentence_id[:sentence_size]

    def loader(self, sentences1, sentences2, label, batch_size=64, sentence_size=32):
        size = min([len(sentences1), len(sentences2)])
        for i in range(0, size, batch_size):
            batch_data = []
            batch_label = []
            for j in range(i, i+batch_size):
                if j >= size:
                    break
                batch_data.append([self.padding(self.sentence2id(sentences1[j]), sentence_size), self.padding(self.sentence2id(sentences2[j]), sentence_size)])
                batch_label.append(self.label2id(label[j]))
            yield torch.LongTensor(batch_data), torch.Tensor(batch_label)


def data_set(max_len=-1, batch_size=64):
    data = pd.read_csv('data/snli_1.0/snli_1.0_train.txt', sep='\t')
    v1 = data['sentence1'][:max_len]
    v2 = data['sentence2'][:max_len]
    label = data['gold_label'][:max_len]
    print(v1, v2)
    vocabulary = Vocab(v1, v2)
    return vocabulary.loader(v1, v2, label, batch_size), vocabulary.vocab_size


if __name__ == '__main__':
    data, size = data_set(3)

