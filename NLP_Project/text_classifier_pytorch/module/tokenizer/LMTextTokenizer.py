

import pickle as pkl



class LMTextTokenizer(object):
    
    def __init__(self, tokenizer):
        
        self.tokenizer = tokenizer
        self.cls_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.unk_token_id = tokenizer.unk_token_id
        # self.convert_tokens_to_ids = ''
        self.load()
        


    def load(self):
        """
        读取分词器
        """
        self.token2index = self.tokenizer.vocab
        self.index2token = { i:x for x,i in self.token2index.items()}


    def tokenizer(self, text):
        """
        分词，按字分词
        """
        token = self.tokenizer(text, return_tensors="pt")
        return token


    def get_special_tokens(self):
        """
        获取特殊字符
        """
        target_ids = [self.cls_token_id, self.pad_token_id, self.sep_token_id, self.unk_token_id]
        target = [self.index2token.get(x, '') for x in target_ids]
        target = [ x for x in target if x]
        return target
    
    