
import os
from posixpath import sep
import time
import random
import logging
import math
import numpy as np
import pandas as pd
import torch
from apex import amp
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig, AutoConfig
# from model.BertForMaskedLM import BertForMaskedLM
from sklearn import metrics


from Config import Config
from utils.progressbar import ProgressBar
from module.ModelMap import map_model, map_tokenizer
from module.tokenizer.LMTextTokenizer import LMTextTokenizer




class Predictor(object):
    
    def __init__(self, config):
        self.config = config
        # self.test_loader = test_loader
        self.device = torch.device(self.config.device)
        # 加载模型
        self.load_label()
        self.load_tokenizer()
        self.load_model()
        
    
    def load_label(self):
        """
        读取标签
        """
        print('loading tag file ...')
        path_label = os.path.join(self.config.path_datasets, 'class.txt')
        self.label = [ x.strip() for x in open(path_label, 'r', encoding='utf8').readlines()]
        self.label2ids = {x:i for i,x in enumerate(self.label)}
        self.ids2label = {i:x for i,x in enumerate(self.label)}
    
    
    def load_tokenizer(self):
        """
        读取分词器
        """
        print('loading tokenizer config ...')
        tokenizer = map_tokenizer(self.config.model_name)
        if not tokenizer:
            print('toknizer {} is null, please check your model name.'.format(self.config.model_name))
        
        if 'Text' in self.config.model_name or 'Transformer' in self.config.model_name:
            path_token = os.path.join(self.config.path_datasets, 'vocab.pkl')
            self.tokenizer = tokenizer()
            # 若存在词表，则直接读取
            if os.path.exists(path_token):
                self.tokenizer.load(path_token)
            else:
                # 否则读取训练数据，并创建词表
                print('vacob file not exist: {}'.format(path_token))
        else:
            tokenizer = tokenizer.from_pretrained(self.config.initial_pretrain_tokenizer)
            self.tokenizer = LMTextTokenizer(tokenizer)
    
    
    def load_model(self):
        """
        加载模型及初始化模型参数
        """
        print('loading model...%s' %self.config.model_name)
        self.model = map_model(self.config.model_name)
        if not self.model:
            print('model {} is null, please check your model name.'.format(self.config.model_name))
        
        if 'Text' in self.config.model_name or 'Transformer' in self.config.model_name:
            path_model = os.path.join(self.config.path_model_save, 'step_best/pytorch_model.bin')
            if not os.path.exists(path_model):
                print('model checkpoint file not exist: {}'.format(path_model))
                return 
            self.model.load_state_dict(torch.load(path_model))
        else:
            # 模型路径
            path_model = os.path.join(self.config.path_model_save, 'step_best/')
            if not os.path.exists(path_model):
                print('model checkpoint file not exist: {}'.format(path_model))
                return 
            path_config = os.path.join(path_model, 'config.json')
            model_config = AutoConfig.from_pretrained(path_config)   #, num_labels=len(self.label)
            self.model = self.model.from_pretrained(path_model, config=model_config)    
        # 将模型加载到CPU/GPU
        self.model.to(self.device)
        self.model.eval()
    
    
    def predict(self, test_loader):
        """
        预测
        """
        print('predict start')        

        # 混合精度
        if self.config.fp16:
            self.model = amp.initialize(self.model, opt_level='O3')

        # 初始化指标计算
        progress_bar = ProgressBar(n_total=len(test_loader), desc='Predict')
        src = []
        label = np.array([], dtype=int)
        pred = np.array([], dtype=int)
        for i, batch in enumerate(test_loader):
            # 推断
            batch = {k:v.to(self.config.device) for k,v in batch.items()}
            with torch.no_grad():
                output = self.model(**batch)
            output = output[0]
            # 输入文本转换
            input_ids = batch['input_ids'].cpu().numpy()
            tmp_src_string = self.ids2string(input_ids)
            
            # 获取标签
            tmp_pred = torch.max(output, 1)[1].cpu().numpy()
            tmp_label = batch['label'].cpu().numpy()
            # 添加到总列表
            src.extend(tmp_src_string)
            label = np.append(label, tmp_label)
            pred = np.append(pred, tmp_pred)
            progress_bar(i, {})

        # 计算指标
        # report = metrics.classification_report(label, pred, target_names=self.label, digits=4)
        # confusion = metrics.confusion_matrix(label, pred)
        # print('Evaluate Classifier Performance')
        # print(report)
        
        # 保存
        data = {'src':src, 'label':label, 'pred':pred}
        data = pd.DataFrame(data)
        if not os.path.exists(self.config.path_output):
            os.mkdir(self.config.path_output)
        path_output = os.path.join(self.config.path_output, 'pred_data.csv')
        data.to_csv(path_output, sep='\t', index=False)
        print('predict result save: {}'.format(path_output))



    def ids2string(self, input_ids):
        """
        将模型输出转换成中文
        """
        # 获取特殊字符
        special_tokens = self.tokenizer.get_special_tokens()
        src = []
        for line in input_ids:
            # 分开是否是预训练语言
            if self.config.model_name in self.config.lm_model_list:
                src_line = self.tokenizer.tokenizer.convert_ids_to_tokens(line)
                # 过滤特殊字符
                src_line = [x for x in src_line if x not in special_tokens]
                src_line = ' '.join(src_line)
            else:
                src_line = ''
                for x in line:
                    tmp_x = self.tokenizer.index2token.get(x, '')
                    # 跳过特殊字符
                    if tmp_x not in special_tokens:
                        src_line += tmp_x
            src.append(src_line)
        return src
        
        
