# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         n_grim
# Description:  
# Author:       Laity
# Date:         2021/9/23
# ---------------------------------------------
# 一般n-gram中的n取2或者3, 这里取2为例
ngram_range = 3

def create_ngram_set(input_list):
    """
    description: 从数值列表中提取所有的n-gram特征
    :param input_list: 输入的数值列表, 可以看作是词汇映射后的列表,
                       里面每个数字的取值范围为[1, 25000]
    :return: n-gram特征组成的集合
    """
    return set(zip(*[input_list[i:] for i in range(ngram_range)]))


input_list = [1, 4, 9, 4, 1, 4]
print(create_ngram_set(input_list))

# zip用法 压缩列表
a = [1, 3, 9]
b = [2, 4, 6, 8]
print(list(zip(a, b)))

# *号拆解列表
d = [input_list[i:] for i in range(ngram_range)]
print(d, '\n', *d)