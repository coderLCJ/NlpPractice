# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         demo_0
# Description:  
# Author:       Laity
# Date:         2021/11/9
# ---------------------------------------------
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import os

data_path = "E:/DESKTOP/Github/DATA/TRAIN_1"
train = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t')
test = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t')

ngram_vectorizer = CountVectorizer(ngram_range=(1, 1))
lr = LogisticRegression()
params = {'C': [0.5, 0.8, 1.0], 'penalty': ['l1', 'l2']}
skf = StratifiedKFold(n_splits=3)
gsv = GridSearchCV(lr, params, cv=skf)
pipeline = Pipeline([("ngram", ngram_vectorizer),
                     ("lr", lr)
                     ])
X = train['Phrase']
y = train['Sentiment']
pipeline.fit(X, y)
test['Sentiment'] = pipeline.predict(test['Phrase'])
test[['Sentiment', 'PhraseId']].set_index('PhraseId').to_csv('sklearn_based_lr.csv')

# 0.587