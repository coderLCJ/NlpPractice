# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         reg_squence
# Description:  
# Author:       Laity
# Date:         2021/9/23
# ---------------------------------------------
from keras.preprocessing import sequence

# cutlen根据数据分析中句子长度分布，覆盖90%左右语料的最短长度.
# 这里假定cutlen为10
cutlen = 10

def padding(x_train):
    """
    description: 对输入文本张量进行长度规范
    :param x_train: 文本的张量表示, 形如: [[1, 32, 32, 61], [2, 54, 21, 7, 19]]
    :return: 进行截断补齐后的文本张量表示
    """
    # 使用sequence.pad_sequences即可完成
    return sequence.pad_sequences(x_train, cutlen)


x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
           [2, 32, 1, 23, 1]]
train = [list(range(10)), list(range(6))]
res = padding(train)
print(res)