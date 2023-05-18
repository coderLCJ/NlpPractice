# SPACES
端到端的长文本摘要模型（法研杯2020司法摘要赛道）。

参考原博客介绍：https://kexue.fm/archives/8046

## 框架

```
├── README.md
├── datasets：数据集
│   ├── README.md
│   ├── train.json
│   ├── user_dict.txt：补充词典
│   └── user_dict_2.txt：补充词典
├── extract_convert.py：数据转换，构建抽取模型训练集
├── extract_vectorize.py：将句子向量化
├── extract_model.py：训练抽取模型
├── final.py：预测
├── seq2seq_convert.py：数据转换，抽取重点句
├── seq2seq_model.py：训练生成模型
└── snippets.py：函数和参数集合
```



## 运行

实验环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.9.7

(如果是Windows，请用bert4keras>=0.9.8)

首先请在`snippets.py`中修改相关路径配置，然后再执行下述代码。

训练代码：
```bash
#! /bin/bash

python extract_convert.py
python extract_vectorize.py

for ((i=0; i<15; i++));
    do
        python extract_model.py $i
    done

python seq2seq_convert.py
python seq2seq_model.py
```

预测代码
```python
from final import *
summary = predict(text, topk=3)
print(summary)
```

## 链接

- 博客：https://kexue.fm
- 追一：https://zhuiyi.ai/
- 预训练模型：https://github.com/ZhuiyiTechnology/pretrained-models
- WoBERT：https://github.com/ZhuiyiTechnology/WoBERT
