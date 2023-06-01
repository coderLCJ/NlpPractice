
import re
import os
import random
import math
import numpy as np
import pandas as pd
import pickle as pkl
import torch
# from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from utils.IOOption import open_file, write_text, write_file

from module.ModelMap import map_tokenizer
from module.tokenizer.TextTokenizer import TextTokenizer
from module.tokenizer.LMTextTokenizer import LMTextTokenizer





class DataManager(object):
    
    def __init__(self, config):
        
        self.config = config
        self.init_gpu_config()          # 初始化GPU配置
        self.load_label()               # 读取标签
        self.load_tokenizer()           # 读取tokenizer分词模型
    
    
    def init_gpu_config(self):
        """
        初始化GPU并行配置
        """
        print('loading GPU config ...')
        if self.config.mode == 'train' and torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(backend='nccl', 
                                                 init_method=self.config.init_method,
                                                 rank=0, 
                                                 world_size=self.config.world_size)
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    
    
    def load_label(self):
        """
        读取标签
        """
        print('loading tag file ...')
        path_label = os.path.join(self.config.path_datasets, 'class.txt')
        label = [ x.strip() for x in open(path_label, 'r', encoding='utf8').readlines()]
        self.label2ids = {x:i for i,x in enumerate(label)}
        self.ids2label = {i:x for i,x in enumerate(label)}
        
    
    def load_tokenizer(self):
        """
        读取分词器
        """
        print('loading tokenizer config ...')
        tokenizer = map_tokenizer(self.config.model_name)
        if not tokenizer:
            print('toknizer {} is null, please check your model name.'.format(self.config.model_name))
        
        if self.config.model_name not in self.config.lm_model_list:
            path_token = os.path.join(self.config.path_datasets, 'vocab.pkl')
            self.tokenizer = tokenizer()
            # 若存在词表，则直接读取
            if os.path.exists(path_token):
                self.tokenizer.load(path_token)
            else:
                # 否则读取训练数据，并创建词表
                path_corpus = os.path.join(self.config.path_datasets, 'train.txt')
                corpus, _ = open_file(path_corpus, sep='\t')
                token2index, _ = self.tokenizer.create(corpus)
                # 标签映射表存到本地
                write_file(token2index, path_token + '.txt')
                pkl.dump(token2index, open(path_token, 'wb'))
                self.tokenizer.load(path_token)
        else:
            tokenizer = tokenizer.from_pretrained(self.config.initial_pretrain_tokenizer)
            self.tokenizer = LMTextTokenizer(tokenizer)
        print('Vocab size: {}'.format(len(self.tokenizer.token2index)))

    
    def get_dataset(self, data_type='train'):
        """
        获取数据集
        """
        file = '{}.txt'.format(data_type)
        dataloader = self.data_process(file)
        return dataloader


    def data_process(self, file_name):
        """
        数据转换
        """
        # 获取数据
        path = os.path.join(self.config.path_datasets, file_name)
        src, tgt = open_file(path, sep='\t')
        dataset = pd.DataFrame({'src':src, 'label':tgt})
        # dataset.to_csv('./data/cache.csv', sep='\t', index=False)
        # dataframe to datasets
        raw_datasets = Dataset.from_pandas(dataset)
        # tokenizer.
        tokenized_datasets = raw_datasets.map(lambda x: self.tokenize_function(x), batched=True)        # 对于样本中每条数据进行数据转换
        # data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)                               # 对数据进行padding
        tokenized_datasets = tokenized_datasets.remove_columns(["src"])                        # 移除不需要的字段
        tokenized_datasets.set_format("torch", columns=["input_ids","attention_mask","label"])   
        # 转换成DataLoader类
        sampler = RandomSampler(tokenized_datasets) if not torch.cuda.device_count() > 1 else DistributedSampler(tokenized_datasets)
        dataloader = DataLoader(tokenized_datasets, sampler=sampler, batch_size=self.config.batch_size)     #, collate_fn=data_collator

        return dataloader


    def tokenize_function(self, example):
        """
        数据转换
        """
        # 分词
        token = {}
        # src = [self.tokenizer.convert_tokens_to_ids(x) for x in example["src"]]
        src_origin = [self.tokenizer.tokenizer(x) for x in example["src"]]
        src = [ x['input_ids'] for x in src_origin ]
        attention_mask = [ x['attention_mask'] for x in src_origin ]
        # paddding
        src = [self.padding(x) for x in src]
        attention_mask = [self.padding_attention(x) for x in attention_mask]
        label = [ int(x) for x in example["label"]]
        # 添加标签到样本中
        token = {
            'input_ids':src,
            'attention_mask':attention_mask,
            'label':label
        }
        return token



    def padding(self, src):
        """
        padding
        """
        # 裁剪
        if len(src) > self.config.sen_max_length:
            src = src[:self.config.sen_max_length]
        # padding
        pad_size = self.config.sen_max_length-len(src)       # 待padding的长度
        # 添加cls/pad/sep特殊字符
        # src = [self.tokenizer.cls_token_id] + src + [self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id]*pad_size
        src = src + [self.tokenizer.pad_token_id]*pad_size
        assert len(src) == self.config.sen_max_length, 'input no equal {}'.format(self.config.sen_max_length)
        return src


    def padding_attention(self, attention_mask):
        """
        padding attention mask
        """
        # 裁剪
        if len(attention_mask) > self.config.sen_max_length:
            attention_mask = attention_mask[:self.config.sen_max_length]
        # padding
        pad_size = self.config.sen_max_length-len(attention_mask)       # 待padding的长度
        # 添加cls/pad/sep特殊字符
        attention_mask = attention_mask + [0]*pad_size
        assert len(attention_mask) == self.config.sen_max_length, 'input no equal {}'.format(self.config.sen_max_length)
        return attention_mask

