#! /usr/bin/env python
# -*- coding: utf-8 -*-
# author: laity
# date: 2023-05-26
import os

class Model:
    def __init__(self):
        pass
    
    def predict(self, inputs=None):
        # 调用模型
        # errCode, errMsg, predictResults = model(inputs)
        errCode = 0
        errMsg = 'Hello world' 
        predictResults = {'BERT': 'Hello World, I am BERT!'} 
        
        # 返回结果 
        return errCode, errMsg, predictResults