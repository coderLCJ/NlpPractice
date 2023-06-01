# Text_Classifier_Pytorch

## Info
基于Pytorch的文本分类框架。

同时支持中英文的数据集的文本分类任务。


## Model
- 非预训练类模型：
    - FastText
    - TextCNN
    - TextRNN
    - TextRCNN
    - Transformer
- 预训练类模型
    - Bert
    - Albert
    - Roberta
    - Distilbert
    - Electra
    - XLNet                                  


## Trianing Mode Support

- 支持中英文语料训练
    - 支持中英文的文本分类任务。
- 支持多种模型使用
    - 配置文件`Config.py`中的变量`model_name`表示模型名称，可以更改成你想要加载的模型名称。
    - 若是预训练类的模型，如Bert等，需要同步修改变量`initial_pretrain_model`和`initial_pretrain_tokenizer`，修改为你想要加载的预训练参数。
- 混合精度训练
    - 用于提升训练过程效率，缩短训练时间。
    - 配置文件`Config.py`中的变量`fp16`值改为`True`。
- GPU多卡训练
    - 用于分布式训练，支持单机单卡、多卡训练。
    - 配置文件`Config.py`中的变量`cuda_visible_devices`用于设置可见的GPU卡号，多卡情况下用`,`间隔开。
- 对抗训练
    - 在模型embedding层增加扰动，使模型学习对抗扰动，提升表现，需要额外增加训练时间。
    - 配置文件`Config.py`中的变量`adv_option`用于设置可见的对抗模式，目前支持FGM/PGD。
- 对比学习
    - 用于增强模型语义特征提取能力，借鉴Rdrop和SimCSE的思想，目前支持KL loss和InfoNCE两种损失。
    - 配置文件`Config.py`中的变量`cl_option`设置为`True`则表示开启对比学习模式，`cl_method`用于设置计算对比损失的方法。



## Datasets
* **THUCNews**
    * 来自：https://github.com/649453932/Chinese-Text-Classification-Pytorch
    * 关于THUCNews的的数据。
    * 数据分为10个类标签类别，分别为：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐

* **加入自己的数据集**
    * 可使用本项目的处理方式，将数据集切分为3部分：train/valid/test，其中token和label之间用制表符`\t`分割。
    * 在 ./dataset 目录下新建一个文件夹，并把3个数据文件放置新建文件夹下。

* **数据集示例**
    * 以数据集THUCNews为栗子，文本和标签使用空格隔开，采用以下形式存储：
    ```
        午评沪指涨0.78%逼近2800 汽车家电农业领涨	2
        卡佩罗：告诉你德国脚生猛的原因 不希望英德战踢点球	7
    ```


## Experiments

说明：预训练模型基于transformers框架，如若想要替换成其他预训练参数，可以查看[transformers官方网站](https://huggingface.co/models)。

| 模型名称 | MicroF1 | LearningRate | 预训练参数 |
| :-----| :---- | :---- | :---- |
| FastText | 0.8926 | 1e-3 | - |
| TextCNN | 0.9009 | 1e-3 | - |
| TextRNN | 0.9080 | 1e-3 | - |
| TextRCNN | 0.9142 | 1e-3 | - |
| Tramsformer(2 layer) | 0.8849 | 1e-3 | - |
| Albert | 0.9124 | 2e-5 | [voidful/albert_chinese_tiny](https://huggingface.co/voidful/albert_chinese_tiny) |
| Distilbert | 0.9209 | 2e-5 | [Geotrend/distilbert-base-zh-cased](https://huggingface.co/Geotrend/distilbert-base-zh-cased) |
| Bert | 0.9401 | 2e-5 | [bert-base-chinese](https://huggingface.co/bert-base-chinese) |
| Roberta | 0.9448 | 2e-5 | [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) |
| Electra | 0.9377 | 2e-5 | [hfl/chinese-electra-base-discriminator](https://huggingface.co/hfl/chinese-electra-base-discriminator) |
| XLNet | 0.9051 | 2e-5 | 无参数初始化 |





## Requirement
Python使用的是3.6.X版本，其他依赖模块如下：
```
    numpy==1.19.2
    pandas==1.1.5
    scikit_learn==1.0.2
    torch==1.8.0
    tqdm==4.62.3
    transformers==4.15.0
    apex==0.1
```

除了`apex`需要额外安装（参考官网：https://github.com/NVIDIA/apex
），其他模块可通过以下命令安装依赖包
```
    pip install -r requirement.txt
```


## Get Started
### 1. 训练
准备好训练数据后，终端可运行命令
```
    python3 main.py
```
### 2 测试评估
加载已训练好的模型，并使用valid set作模型测试，输出文件到 ./dataset/${your_dataset}/output/output.txt 目录下。

需要修改Config文件中的变量值`mode = 'test'`，并保存。

终端可运行命令
```
    python3 main.py
```


## Reference

[Github:transformers] https://github.com/huggingface/transformers

[Paper:Bert] https://arxiv.org/abs/1810.04805

[Paper:RDrop] https://arxiv.org/abs/2106.14448

[Paper:SimCSE] https://arxiv.org/abs/2104.08821
