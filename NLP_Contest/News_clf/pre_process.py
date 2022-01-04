# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         pre_process
# Description:  
# Author:       Laity
# Date:         2022/1/4
# ---------------------------------------------
from time import sleep

import pandas as pd

file = pd.read_csv('E:/DESKTOP/Github/DATA/News_clf/train_set.csv', sep='\\t', engine='python')
# print(file)
# print(file['text'])

# max_num = -1
# max_len = -1
# count = 0
# for i in file['text']:
#     word = i.split()
#     t = max(int(j) for j in word)
#     max_num = t if t > max_num else max_num
#     if len(word) > max_len:
#         max_len = len(word)
#         print(max_len, count)
#         print('====================')
#     count += 1
#
# print(count)
# print(max_num)
# print(max_len)

print(len(file['text'][72749].split(' ')))
'''
max_len = 57921 => 58000
max_num = 7549  


1057 0
====================
1570 3
====================
3900 36
====================
7125 71
====================
10018 419
====================
10065 2202
====================
12211 2491
====================
13434 5913
====================
15569 6570
====================
25662 7755
====================
44665 8462
====================
53527 72749
====================
'''
