

import os
import time
import numpy as np
import torch
import logging
from Config import Config
from module.DataManager import DataManager
from module.Trainer import Trainer
from module.Predictor import Predictor



if __name__ == '__main__':


    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    # 设置随机种子，保证结果每次结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    start_time = time.time()

    # 数据处理
    print('read data...')
    dm = DataManager(config)

    # 模式
    if config.mode == 'train':
        # 获取数据
        print('data process...')
        train_loader = dm.get_dataset(data_type='train')
        valid_loader = dm.get_dataset(data_type='dev')
        test_loader = dm.get_dataset(data_type='test')
        # 训练
        trainer = Trainer(config, train_loader, valid_loader, test_loader)
        trainer.train()
    elif config.mode == 'test':
        # 测试
        test_loader = dm.get_dataset(data_type='test')
        predictor = Predictor(config)
        predictor.predict(test_loader)
    else:
        print("no task going on!")
        print("you can use one of the following lists to replace the valible of Config.py. ['train', 'test', 'valid'] !")
        