from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import numpy as np
import model_train
import torch.nn

import torch
import torch.nn as nn

text = ['hello World hello Beijing']
ct = CountVectorizer()
vec = ct.fit_transform(text)
print(ct.vocabulary_)
bag = ct.vocabulary_
for i in text[0].split():
    print(bag[i])