# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         main
# Description:  酒店评论情感分析
# Author:       Laity
# Date:         2021/9/22
# ---------------------------------------------
# 导入必备工具包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# 设置显示风格
plt.style.use('fivethirtyeight')

def makeCsv():
    train_size = 900
    size = 1000
    train_data = []
    valid_data = []
    ft = open('../comment/train.csv', 'w', encoding='utf8')
    fv = open('../comment/dev.csv', 'w', encoding='utf8')

    for i in range(size):
        neg = '../comment/neg/neg.' + str(i) + '.txt'
        pos = '../comment/pos/pos.' + str(i) + '.txt'
        neg_data = open(neg, errors='ignore').read().split()[0] + '\t0\n'
        pos_data = open(pos, errors='ignore').read().split()[0] + '\t1\n'
        if i < train_size:
            ft.write(neg_data)
            ft.write(pos_data)
        else:
            fv.write(neg_data)
            fv.write(pos_data)


# 分别读取训练tsv和验证tsv
# makeCsv()
train_data = pd.read_csv("../comment/train.csv", sep="\t")
valid_data = pd.read_csv("../comment/dev.csv", sep="\t")


# 获得训练数据标签数量分布

sns.countplot(x='label', data=train_data)
plt.title("train_data")
plt.show()

# 获取验证数据标签数量分布
sns.countplot("label", data=valid_data)
plt.title("valid_data")
plt.show()
