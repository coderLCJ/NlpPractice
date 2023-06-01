

import pickle as pkl
from utils.IOOption import open_file, write_file



class TextTokenizer(object):
    
    def __init__(self):
        
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.sep_token = '[SEP]'
        self.unk_token = '[UNK]'
        self.convert_tokens_to_ids = ''
        

    def load(self, path):
        """
        读取分词器
        """
        self.token2index = pkl.load(open(path, 'rb'))
        self.index2token = { i:x for x,i in self.token2index.items()}
        self.cls_token_id = self.token2index.get(self.cls_token)
        self.pad_token_id = self.token2index.get(self.pad_token)
        self.sep_token_id = self.token2index.get(self.sep_token)
        self.unk_token_id = self.token2index.get(self.unk_token)
        

    def create(self, corpus):
        """
        创建分词字典，获取训练集词表
        """
        # 按字分词
        words = [w for line in corpus for w in line if w != '']
        words = list(set(words))
        words = sorted(words, reverse=False)
        # 创建索引
        token2index = {x:i for i,x in enumerate(words)}
        index2token = {i:x for i,x in enumerate(words)}

        # 添加特殊字符
        if self.pad_token not in token2index.keys():
            index2token[len(token2index)] = self.pad_token
            token2index[self.pad_token] = len(token2index)
        if self.unk_token not in token2index.keys():
            index2token[len(token2index)] = self.unk_token
            token2index[self.unk_token] = len(token2index)
        if self.cls_token not in token2index.keys():
            index2token[len(token2index)] = self.cls_token
            token2index[self.cls_token] = len(token2index)
        if self.sep_token not in token2index.keys():
            index2token[len(token2index)] = self.sep_token
            token2index[self.sep_token] = len(token2index)
        self.token2index = token2index
        self.index2token = index2token
        return token2index, index2token


    def tokenizer(self, text):
        """
        分词，按字分词
        """
        tokens = [ x for x in text]
        input_ids = [self.token2index.get(x, self.unk_token_id) for x in tokens]
        attention_mask = [0]*len(input_ids)
        token = {
            'input_ids' : input_ids,
            'attention_mask' : attention_mask
        }
        return token


    def get_special_tokens(self):
        """
        获取特殊字符
        """
        target_ids = [self.cls_token_id, self.pad_token_id, self.sep_token_id, self.unk_token_id]
        target = [self.index2token.get(x, '') for x in target_ids]
        target = [ x for x in target if x]
        return target