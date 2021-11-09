import numpy as np
import pandas as pd
# from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

batch_size = 128
nb_classes = 11
nb_epoch = 300

# load data
train_df = pd.read_csv('E:/DESKTOP/Github/DATA/TRAIN_1/train.tsv', sep='\t', header=0)
test_df = pd.read_csv('E:/DESKTOP/Github/DATA/TRAIN_1/test.tsv', sep='\t', header=0)

raw_docs_train = train_df['Phrase'].values
raw_docs_test = test_df['Phrase'].values
sentiment_train = train_df['Sentiment'].values
num_labels = len(np.unique(sentiment_train))

# text pre-processing
stop_words = set(stopwords.words('english'))  # 停用词
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
stemmer = SnowballStemmer('english')  # 词干还原器 自动转换小写

print("pre-processing train docs...")
processed_docs_train = []
for doc in raw_docs_train:
    tokens = word_tokenize(doc)  # 分词
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    processed_docs_train.append(stemmed)
print(len(processed_docs_train))

print("pre-processing test docs...")
processed_docs_test = []
for doc in raw_docs_test:
    tokens = word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    processed_docs_test.append(stemmed)

processed_docs_all = np.concatenate((processed_docs_train, processed_docs_test), axis=0)
print(len(processed_docs_all))
