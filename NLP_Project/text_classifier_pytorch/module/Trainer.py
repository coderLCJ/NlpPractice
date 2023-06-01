
import os
from posixpath import sep
import time
import random
import logging
import math
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import torch.nn as nn
from apex import amp
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig, AutoConfig
# from model.BertForMaskedLM import BertForMaskedLM


import torch.nn.functional as F
from sklearn import metrics

from utils.progressbar import ProgressBar
from module.optimal.adversarial import FGM,PGD
from module.ModelMap import map_model, map_config, map_tokenizer
from module.LossManager import LossManager



class Trainer(object):
    
    def __init__(self, config, train_loader, valid_loader, test_loader):
        self.config = config
        # 设置GPU环境
        self.device = torch.device(self.config.device)
        # 加载数据集
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        # 加载标签
        self.load_label()
        # 加载模型
        self.load_tokenizer()
        self.load_model()
        # 加载loss计算类
        self.loss_manager = LossManager(loss_type=config.loss_type, cl_option=config.cl_option, loss_cl_type=config.cl_method)



    def load_label(self):
        """
        读取标签
        """
        path_label = os.path.join(self.config.path_datasets, 'class.txt')
        self.label = [ x.strip() for x in open(path_label, 'r', encoding='utf8').readlines()]
        self.label2ids = {x:i for i,x in enumerate(self.label)}
        self.ids2label = {i:x for i,x in enumerate(self.label)}


    def load_tokenizer(self):
        """
        读取分词器
        """
        self.tokenizer = map_tokenizer(self.config.model_name)


    def load_model(self):
        """
        加载模型及初始化模型参数
        """
        # 读取模型
        print('loading model...%s' %self.config.model_name)
        self.model = map_model(self.config.model_name)
        if not self.model:
            print('model {} is null, please check your model name.'.format(self.config.model_name))
        
        if self.config.model_name not in self.config.lm_model_list:
            # self.model = map_model(self.config.model_name)
            model_config = map_config(self.config.model_name)(self.config)
            self.model = self.model(model_config)
            # 重新初始化模型参数
            self.init_network()
        else:
            # self.tokenizer = map_tokenizer(self.config.model_name).from_pretrained(self.config.model_pretrain_online_checkpoint)
            # self.tokenizer.save_pretrained(self.config.path_tokenizer)
            # self.func_index2token = self.tokenizer.convert_ids_to_tokens
            # 加载预训练模型
            model_config = AutoConfig.from_pretrained(self.config.initial_pretrain_model, num_labels=len(self.label))   #, num_labels=len(self.label2ids)
            self.model = self.model.from_pretrained(self.config.initial_pretrain_model, config=model_config)    
        # 将模型加载到CPU/GPU
        self.model.to(self.device)

    
    def init_network(self, method='xavier', exclude='embedding', seed=123):
        """
        # 权重初始化，默认xavier
        """
        for name, w in self.model.named_parameters():
            if exclude not in name:
                if 'weight' in name:
                    if method == 'xavier':
                        if 'transformer' in name:
                            nn.init.uniform_(w, -0.1, 0.1)
                        else:
                            nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass


    def train(self):
        """
            预训练模型
        """
        # weight decay
        # bert_parameters = self.model.bert.named_parameters()
        # start_parameters = self.model.start_fc.named_parameters()
        # end_parameters = self.model.end_fc.named_parameters()
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.01, 'lr': self.config.learning_rate},
        #     {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        #         , 'lr': self.config.learning_rate},
        #     {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.01, 'lr': 0.001},
        #     {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        #         , 'lr': 0.001},
        #     {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.01, 'lr': 0.001},
        #     {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        #         , 'lr': 0.001}]
        # step_total = self.config.num_epochs * len(self.train_loader) * self.config.batch_size
        # # step_total = 640 #len(train_ld)*config.batch_size // config.num_epochs
        # warmup_steps = int(step_total * self.config.num_warmup_steps)
        # self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=1e-8)
        # self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
        #                                             num_training_steps=step_total)
        
        # 定义优化器配置
        # num_training_steps = self.config.num_epochs * len(self.train_loader)
        # 总的训练次数
        step_total = self.config.num_epochs * len(self.train_loader) * self.config.batch_size
        # warm up的次数
        warmup_steps = int(step_total * self.config.num_warmup_steps)
        if self.config.model_name not in self.config.lm_model_list:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=step_total
            )
            # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
            #                                             num_training_steps=t_total)
        
        # 混合精度训练
        if self.config.fp16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.config.fp16_opt_level)
        # 分布式训练
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
        # 对抗训练
        if self.config.adv_option == 'FGM':
            self.fgm = FGM(self.model, emb_name=self.config.adv_name, epsilon=self.config.adv_epsilon)
        if self.config.adv_option == 'PGD':
            self.pgd = PGD(self.model, emb_name=self.config.adv_name, epsilon=self.config.adv_epsilon)

        # Train!
        print("\n>>>>>>>> Running training >>>>>>>>")
        print("  Num examples = %d" %(len(self.train_loader)*self.config.batch_size))
        print("  Num Epochs = %d" %self.config.num_epochs)
        print("  Batch size per GPU = %d"%self.config.batch_size)
        print("  GPU ids = %s" %self.config.cuda_visible_devices)
        print("  Total step = %d" %step_total)
        print("  Warm up step = %d" %warmup_steps)
        print("  FP16 Option = %s" %self.config.fp16)
        print(">>>>>>>> Running training >>>>>>>>\n")
        
        print(">>>>>>>> Model Structure >>>>>>>>")
        for name,parameters in self.model.named_parameters():
            print(name,':',parameters.size())
        print(">>>>>>>> Model Structure >>>>>>>>\n")

        # step_total = config.num_epochs * len(train_ld)
        step_current = 0
        f1_best = 0
        for epoch in range(self.config.num_epochs):
            progress_bar = ProgressBar(n_total=len(self.train_loader), desc='Training epoch:{0}'.format(epoch))
            for i, batch in enumerate(self.train_loader):
                # 模型推断及计算损失
                self.model.train()
                loss = self.step(batch)
                progress_bar(i, {'loss': loss.item()})
                # progress_bar(i, {'loss': loss.item(),'loss_ce': loss_ce.item(),'loss_cl': loss_nce.item()})
                step_current += 1
                # 模型保存
                if step_current%self.config.step_save==0 and step_current>0:
                    # 模型评估
                    f1_eval = self.evaluate(self.valid_loader)
                    # 模型保存
                    f1_best = self.save_checkpoint(step_current, f1_eval, f1_best)
            print('\nEpoch:{}  Iter:{}/{}  loss:{:.4f}\n'.format(epoch, step_current, step_total, loss.item()))
        self.evaluate(self.test_loader, print_table=True)
    
    

    def step(self, batch):
        """
        每一个batch的训练过程
        """
        
        # 数据操作
        batch = {k:v.to(self.device) for k,v in batch.items()}
        target = batch['label']
        # 模型输入&输出
        outputs = self.model(**batch)
        output, hidden_emb = outputs
        # 对比学习
        if self.config.cl_option:
            # 重新获取一次模型输出
            outputs_etx = self.model(**batch)
            _, hidden_emb_etx = outputs_etx
            loss = self.loss_manager.compute(output, target, hidden_emb, hidden_emb_etx, alpha=self.config.cl_loss_weight)
        else:
            loss = self.loss_manager.compute(output, target)
        # 反向传播
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        if self.config.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # 对抗训练
        self.attack_train(batch)
        # 梯度操作
        self.optimizer.step()
        if self.config.model_name in self.config.lm_model_list:
            self.lr_scheduler.step()
        self.model.zero_grad()
        # self.optimizer.zero_grad()
        return loss


    def attack_train(self, batch):
        """
        对抗训练
        """
        # FGM
        if self.config.adv_option == 'FGM':
            self.fgm.attack()
            output = self.model(**batch)[0]
            loss_adv = self.loss_manager.compute(output, batch['label'])
            if torch.cuda.device_count() > 1:
                loss_adv = loss_adv.mean()
            loss_adv.backward()
            self.fgm.restore()
        # PGD
        if self.config.adv_option == 'PGD':
            self.pgd.backup_grad()
            K = 3
            for t in range(K):
                self.pgd.attack(is_first_attack=(t==0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    self.model.zero_grad()
                else:
                    self.pgd.restore_grad()
                output = self.model(**batch)[0]
                loss_adv = self.loss_manager.compute(output, batch['label'])
                loss_adv.backward()                      # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            self.pgd.restore()   
            

    def save_checkpoint(self, step_current, f1_eval, f1_best):
        """
        模型保存
        """
        if f1_eval != 0:
            # 保存路径
            path = os.path.join(self.config.path_model_save, 'step_{}'.format(step_current))
            if not os.path.exists(path):
                os.makedirs(path)
            # 保存当前step的模型
            if self.config.model_name not in self.config.lm_model_list:
                path_model = os.path.join(path, 'pytorch_model.bin')
                torch.save(self.model.state_dict(), path_model)
            else:
                model_save = self.model.module if torch.cuda.device_count() > 1 else self.model
                model_save.save_pretrained(path)
            print('Saving model: {}'.format(path))
            # 保存最优的模型
            if f1_eval > f1_best:
                # 创建文件夹
                path = os.path.join(self.config.path_model_save, 'step_best/')
                if not os.path.exists(path):
                    os.makedirs(path)
                # 模型保存
                if self.config.model_name not in self.config.lm_model_list:
                    path_model = os.path.join(path, 'pytorch_model.bin')
                    torch.save(self.model.state_dict(), path_model)
                else:
                    model_save = self.model.module if torch.cuda.device_count() > 1 else self.model
                    model_save.save_pretrained(path)
                f1_best = f1_eval
                print('Saving best model: {}\n'.format(path))
        return f1_best


    def evaluate(self, data, print_table=False):
        """
        模型测试集效果评估
        """
        self.model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        loss_manager = LossManager(loss_type=self.config.loss_type, cl_option=False)
        with torch.no_grad():
            for i, batch in enumerate(data):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                output = self.model(**batch)[0]
                # 计算loss
                # loss = F.cross_entropy(outputs, labels)
                # loss_total += outputx[0]
                target = batch['label']
                loss = loss_manager.compute(output, target)
                loss_total += loss
                # 获取标签
                labels = batch['label'].cpu().numpy()#[:,1:-1]
                predic = torch.max(output, -1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)
        # 计算指标
        acc = metrics.accuracy_score(labels_all, predict_all)
        f1 = metrics.f1_score(labels_all, predict_all, average='micro')
        print('\n>>Eval Set>>:  Loss:{:.4f}  Acc:{}  MicroF1:{:.4f}'.format(loss_total.item(), acc, f1))
        # {'micro', 'macro', 'samples','weighted', 'binary'}
        if print_table:
            # 打印指标
            report = metrics.classification_report(labels_all, predict_all, target_names=self.label, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            print('\nEvaluate Classifier Performance '+'#'*50)
            print(report)
            print('\nConfusion Matrix')
            print(confusion)
            print('#'*60)
            
        return f1
    
    