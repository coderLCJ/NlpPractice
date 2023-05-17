# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         translate
# Description:  
# Author:       Laity
# Date:         2022/3/22
# ---------------------------------------------
import requests

def translate(word):
    # word = str(input("请输入一段要翻译的文字："))
    data = {
        'doctype': 'json',
        'type': 'AUTO',
        'i': word
    }

    url = "http://fanyi.youdao.com/translate"
    r = requests.get(url, params=data)
    result = r.json()['translateResult'][0][0]['tgt']
    data['i'] = result
    r = requests.get(url, params=data)
    result = r.json()['translateResult'][0][0]['tgt']
    return result


def data_enhanced(data):
    i = 0
    for line in data:
        ret = translate(line)
        print('原文 = %s增强 = %s\n' % (line, ret))
        if i > 10:
            break
        i += 1


data = open(r'E:/DESKTOP/论文/代码/生成问句\data/test/seq.in', 'r', encoding='utf8')
data_enhanced(data)
