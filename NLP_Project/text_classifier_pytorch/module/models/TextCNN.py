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
        self.model_name = 'TextCNN'
        path_class = os.path.join(config.path_datasets, 'class.txt')
        self.class_list = [x.strip() for x in open(path_class, encoding='utf-8').readlines()]              # 类别名单
        self.num_classes = len(self.class_list)                         # 类别数
        # embedding config
        file_embedding = 'random'
        path_embedding = os.path.join(config.path_datasets, file_embedding)
        self.embedding_pretrained = torch.tensor(np.load(path_embedding)["embeddings"].astype('float32')) if file_embedding != 'random' else None                                          # 预训练词向量
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        # self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')   # 设备
        # vocab
        path_vocab = os.path.join(config.path_datasets, 'vocab.pkl')
        toekn2index = pkl.load(open(path_vocab, 'rb'))
        self.n_vocab = len(toekn2index.keys())                                                # 词表大小，在运行时赋值
        # model config
        self.dropout = 0.5                                              # 随机失活
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.c = config
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv): 
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, label, attention_mask):
        out = self.embedding(input_ids)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out_drop = self.dropout(out)
        out = self.fc(out_drop)
        return [out, out_drop]
    
        # # 计算loss
        # loss = None
        # if label is not None:
        #     loss_func = CrossEntropyLoss()
        #     # out_softmax = F.softmax(out)
        #     loss = loss_func(out, label)
        #     # loss = F.cross_entropy(out, label)
        # output = (loss, out)
        # return output
