# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2022/3/26
# ---------------------------------------------
import pandas as pd
import copy

def y():
    for i in range(10):
        yield i


def load():
    yield y()



for i in range(3):
    print(i, '=======')
    for x in t:
        print(x)