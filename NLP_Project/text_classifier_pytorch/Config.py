
import os
import random

from module.models.Transformer import Transformer




class Config(object):
        
    # 运行模式
    mode = 'train'
    
    # GPU配置
    cuda_visible_devices = '0'                                  # 可见的GPU
    device = 'cuda:0'                                           # master GPU
    port = str(random.randint(10000,60000))                     # 多卡训练进程间通讯端口
    init_method = 'tcp://localhost:' + port                     # 多卡训练的通讯地址
    world_size = 1                                              # 线程数，默认为1
    
    # 模型选型
    # 基础模型：FastText/TextCNN/TextRNN/TextRCNN/Transformer
    # 语言模型：Bert/Albert/Roberta/Distilbert/Electra/XLNet
    model_name='Bert'                                      
    initial_pretrain_model = 'bert-base-chinese'           # 加载的预训练分词器checkpoint
    initial_pretrain_tokenizer = 'bert-base-chinese'       # 加载的预训练模型checkpoint
    lm_model_list = ['Bert','Albert','Roberta','Distilbert','Electra','XLNet']
    
    # 训练配置
    num_epochs = 30                                             # 迭代次数
    batch_size = 128                                            # 每个批次的大小
    learning_rate = 2e-5                                        # 学习率
    num_warmup_steps = 0.1                                      # warm up步数
    sen_max_length = 32                                         # 句子最长长度
    padding = True                                              # 是否对输入进行padding
    step_save = 1000                                            # 多少步保存一次模型
    loss_type = 'ce'
    
    # 对比学习
    cl_option = True                                            # 是否使用对比学习
    cl_method = 'Rdrop'                                         # Rdrop/InfoNCE
    cl_loss_weight = 0.5                                        # 对比学习loss比例
    # 对抗训练
    adv_option = 'None'                                         # 是否引入对抗训练：none/FGM/PGD
    adv_name = 'word_embeddings'
    adv_epsilon = 1.0
    # 混合精度训练
    fp16 = False
    fp16_opt_level = 'O1'                                   # 训练可选'O1'，测试可选'O3'
    
    # 模型及路径配置
    path_root = os.getcwd()
    path_model_save = os.path.join(path_root, 'checkpoints/')                      # 模型保存路径
    path_datasets = os.path.join(path_root, 'datasets/THUCNews')            # 数据集
    path_log = os.path.join(path_root, 'logs')
    path_output = os.path.join(path_datasets, 'outputs')





