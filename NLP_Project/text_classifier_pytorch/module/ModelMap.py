

from module.models.FastText import FastText, Config as FastTextConfig
from module.models.TextCNN import TextCNN, Config as TextCNNConfig
from module.models.TextRNN import TextRNN, Config as TextRNNConfig
from module.models.TextRCNN import TextRCNN, Config as TextRCNNConfig
from module.models.Transformer import Transformer, Config as TransformerConfig
from module.models.Bert import Bert
from module.models.Albert import Albert
from module.models.Roberta import Roberta
from module.models.Distilbert import Distilbert
from module.models.Electra import Electra
from module.models.XLNet import XLNet
from transformers import AutoTokenizer





from transformers import AlbertTokenizer, BertTokenizer, DistilBertTokenizer, RobertaTokenizer, ElectraTokenizer, XLNetTokenizer
from module.tokenizer.TextTokenizer import TextTokenizer


def map_model(model_name):
    """
    模型映射函数
    """
    dic = {
        'FastText' : FastText,
        'TextCNN' : TextCNN,
        'TextRNN' : TextRNN,
        'TextRCNN' : TextRCNN,
        'Transformer' : Transformer,
        'Bert' : Bert,
        'Albert' : Albert,
        'Roberta' : Roberta,
        'Distilbert' : Distilbert,
        'Electra' : Electra,
        'XLNet' : XLNet
    }
    model = dic.get(model_name, None)
    return model
    

def map_tokenizer(model_name):
    """
    分词器映射函数
    """
    dic = {
        'FastText' : TextTokenizer,
        'TextCNN' : TextTokenizer,
        'TextRNN' : TextTokenizer,
        'TextRCNN' : TextTokenizer,
        'Transformer' : TextTokenizer,
        'Bert' : BertTokenizer,
        'Albert' : AutoTokenizer,
        'Roberta' : BertTokenizer,
        'Distilbert' : DistilBertTokenizer,
        'Electra' : AutoTokenizer,
        'XLNet' : AutoTokenizer
    }
    tokenizer = dic.get(model_name, None)
    return tokenizer


def map_config(model_name):
    """
    模型配置映射
    """
    dic = {
        'FastText' : FastTextConfig,
        'TextCNN' : TextCNNConfig,
        'TextRNN' : TextRNNConfig,
        'TextRCNN' : TextRCNNConfig,
        'Transformer' : TransformerConfig
    }
    model = dic.get(model_name, None)
    return model