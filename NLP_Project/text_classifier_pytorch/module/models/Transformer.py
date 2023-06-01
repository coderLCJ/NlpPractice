# coding: UTF-8
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import dropout
import pickle as pkl


class Config(object):

    """配置参数"""
    def __init__(self, config):
        self.model_name = 'Transormer_base'
        path_class = os.path.join(config.path_datasets, 'class.txt')
        self.class_list = [x.strip() for x in open(path_class, encoding='utf-8').readlines()]              # 类别名单
        self.num_classes = len(self.class_list)                         # 类别数
        # embedding config
        file_embedding = 'random'
        path_embedding = os.path.join(config.path_datasets, file_embedding)
        self.embedding_pretrained = torch.tensor(np.load(path_embedding)["embeddings"].astype('float32')) if file_embedding != 'random' else None                                          # 预训练词向量
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 768           # 字向量维度
        # self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')   # 设备
        # vocab
        path_vocab = os.path.join(config.path_datasets, 'vocab.pkl')
        toekn2index = pkl.load(open(path_vocab, 'rb'))
        self.n_vocab = len(toekn2index.keys())                                                # 词表大小，在运行时赋值
        # model config
        self.dropout = 0.3                                              # 随机失活
        self.nhead = 12
        self.hidden_size = 3072
        self.nlayer = 6
        self.sen_length = config.sen_max_length



class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.c = config
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed, nhead=config.nhead, dim_feedforward=config.hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.nlayer)
        self.pos_encoder = PositionalEncoding(d_model=config.embed, max_len=config.sen_length)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.sen_length, config.num_classes)

    def forward(self, input_ids, label, attention_mask):
        out = self.embedding(input_ids)         # (batch_size, seq_len, emb_size)
        out = out.transpose(0,1)                # (seq_len, batch_size, emb_size)
        out = self.pos_encoder(out)             # (seq_len, batch_size, emb_size)
        out = self.transformer_encoder(out)     # (seq_len, batch_size, emb_size)
        out = out.transpose(0,1)                # (batch_size, seq_len, emb_size)
        out_pool = F.max_pool1d(out, out.size(2)).squeeze(2)
        out = self.fc(out_pool)
        return [out,out_pool]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

