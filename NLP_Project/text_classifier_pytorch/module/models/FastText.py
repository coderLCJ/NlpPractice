# coding: UTF-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from torch.nn import CrossEntropyLoss


class Config(object):

    """配置参数"""
    def __init__(self, config):
        self.model_name = 'FastText'
        path_class = os.path.join(config.path_datasets, 'class.txt')
        self.class_list = [x.strip() for x in open(path_class, encoding='utf-8').readlines()]              # 类别名单
        self.num_classes = len(self.class_list)                         # 类别数
        # embedding config
        file_embedding = 'random'
        path_embedding = os.path.join(config.path_datasets, file_embedding)
        self.embedding_pretrained = torch.tensor(np.load(path_embedding)["embeddings"].astype('float32')) if file_embedding != 'random' else None                                          # 预训练词向量
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度                                    # 预训练词向量
        # self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')   # 设备

        # vocab
        path_vocab = os.path.join(config.path_datasets, 'vocab.pkl')
        toekn2index = pkl.load(open(path_vocab, 'rb'))
        self.n_vocab = len(toekn2index.keys())                                                # 词表大小，在运行时赋值
        # model config
        self.dropout = 0.5                                              # 随机失活
        self.hidden_size = 256     


class FastText(nn.Module):
    def __init__(self, config):
        super(FastText, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, label, attention_mask):

        out = self.embedding(input_ids)                  # size: (batch_size, seq_len, dim)
        out = out.mean(dim=1)                       # size: (batch_size, dim)
        out = self.dropout(out)
        out = self.fc1(out)                         # size: (batch_size, hidden_size)
        out_relu = F.relu(out)
        out = self.fc2(out_relu)                         # size: (batch_size, num_class)
        return [out,out_relu]
        # # 计算loss
        # loss = None
        # if label is not None:
        #     loss_func = CrossEntropyLoss()
        #     # out_softmax = F.softmax(out)
        #     loss = loss_func(out, label)
        #     # loss = F.cross_entropy(out, label)
        # output = (loss, out)
        # return output

