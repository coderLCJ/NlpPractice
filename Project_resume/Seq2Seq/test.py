# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2022/3/22
# ---------------------------------------------
from collections import Counter


class MyClass:
    def __init__(self):
        self.id = [1, 2, 3]
        self.word = ['I', 'love', 'you']

    def __call__(self, idx):
        print('call function')
        for i in range(idx):
            print(i, end=' ')

    def __len__(self):
        print('len function')
        return 1

    def __iter__(self):
        print('\niter')
        for i in range(10):
            yield i


x = [1, 1, 1, 2, 3, 3, 4]
x.extend([2, 4, 5])
print(x)
c = Counter(x)
print(c.most_common())