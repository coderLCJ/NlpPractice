# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         demo0
# Description:  
# Author:       Laity
# Date:         2021/10/19
# ---------------------------------------------
import re

from nltk.util import ngrams

token = 'Thomas Jefferson began building Monticello at Monticello at the age of 26'
pattern = re.compile(r'[-\s.,?:!]+')
token = pattern.split(token)
token = [i for i in token if i not in '-\t\n.,;:?']
print(list(ngrams(token, 3)))