# coding: UTF-8
import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl


class Config(object):

    """配置参数"""
    def __init__(self, config):
        self.model_name = 'TextRNN'
        path_class = os.path.join(config.path_datasets, 'class.txt')
        self.class_list = [x.strip() for x in open(path_class, encoding='utf-8').readlines()]              # 类别名单
        self.num_classes = len(self.class_list)                         # 类别数
        # embedding config
        file_embedding = 'random'
        path_embedding = os.path.join(config.path_datasets, file_embedding)
        self.embedding_pretrained = torch.tensor(np.load(path_embedding)["embeddings"].astype('float32')) if file_embedding != 'random' else None                                          # 预训练词向量
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        # vocab
        path_vocab = os.path.join(config.path_datasets, 'vocab.pkl')
        toekn2index = pkl.load(open(path_vocab, 'rb'))
        self.n_vocab = len(toekn2index.keys())  
        # model config
        self.dropout = 0.5                                              # 随机失活                                      # 学习率
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextRNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, input_ids, label, attention_mask):
        out = self.embedding(input_ids)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out_lstm, _ = self.lstm(out)
        out = self.fc(out_lstm[:, -1, :])  # 句子最后时刻的 hidden state
        return [out, out_lstm]
