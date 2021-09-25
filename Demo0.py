# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         Demo0
# Description:  人名分类器
# Author:       Laity
# Date:         2021/9/25
# ---------------------------------------------
# 从io中导入文件打开方法
from io import open
# 帮助使用正则表达式进行子目录的查询
import glob
import os
# 用于获得常见字母及字符规范化
import string
import unicodedata
# 导入随机工具random
import random
# 导入时间和数学工具包
import time
import math
# 导入torch工具
import torch
# 导入nn准备构建模型
import torch.nn as nn
# 引入制图工具包
import matplotlib.pyplot as plt

# 获取所有常用字符包括字母和常用标点
all_letters = string.ascii_letters + " .,;'"
# 获取常用字符数量
n_letters = len(all_letters)
# print("n_letter:", n_letters, all_letters)

# 关于编码问题我们暂且不去考虑
# 我们认为这个函数的作用就是去掉一些语言中的重音标记
# 如: Ślusàrski ---> Slusarski
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

data_path = "./data/names/"
def readLines(filename):
    """从文件中读取每一行加载到内存中形成列表"""
    # 打开指定文件并读取所有内容, 使用strip()去除两侧空白符, 然后以'\n'进行切分
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # 对应每一个lines列表中的名字进行Ascii转换, 使其规范化.最后返回一个名字列表
    return [unicodeToAscii(line) for line in lines]

# 构建的category_lines形如：{"English":["Lily", "Susan", "Kobe"], "Chinese":["Zhang San", "Xiao Ming"]}
category_lines = {}
# all_categories形如： ["English",...,"Chinese"]
all_categories = []
def createDict():
    # 读取指定路径下的txt文件， 使用glob，path中可以使用正则表达式
    for filename in glob.glob(data_path + '*.txt'):
        # 获取每个文件的文件名, 就是对应的名字类别
        category = os.path.splitext(os.path.basename(filename))[0]
        # 将其逐一装到all_categories列表中
        all_categories.append(category)
        # 然后读取每个文件的内容，形成名字列表
        lines = readLines(filename)
        # 按照对应的类别，将名字列表写入到category_lines字典中
        category_lines[category] = lines

# 改为one-hot编码
def lineToTensor(line):
    code = torch.zeros(len(line), 1, n_letters)
    for index, letter in enumerate(line):
        code[index][0][all_letters.find(letter)] = 1
    return code


createDict()
n_categories = len(category_lines)
# 因为是onehot编码, 输入张量最后一维的尺寸就是n_letters
input_size = n_letters

# 定义隐层的最后一维尺寸大小
n_hidden = 128

# 输出尺寸为语言类别总数n_categories
output_size = n_categories

# num_layer使用默认值, num_layers = 1


input = lineToTensor('B').squeeze(0)
print(input.shape)
input = lineToTensor('B')
print(input.shape)



