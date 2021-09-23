# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         userTokenizer
# Description:  
# Author:       Laity
# Date:         2021/9/22
# ---------------------------------------------
import joblib

j = joblib.load('Tokenizer')

token = '周杰伦'
one_hot = [0] * 6
one_hot[j.texts_to_sequences([token])[0][0]-1] = 1
print('周杰伦的one_hot编码为：', one_hot)
