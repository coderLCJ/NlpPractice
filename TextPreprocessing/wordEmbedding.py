# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         wordEmbedding
# Description:  
# Author:       Laity
# Date:         2021/9/22
# ---------------------------------------------
import torch
import json
import fileinput
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

write = SummaryWriter()


embedded = torch.randn(76, 50)

meta = [line for line in open('data/test.txt', 'r', encoding='utf8').readlines()]
write.add_embedding(embedded, metadata=meta)
write.close()

file_writer = tf.summary.FileWriter('/path/to/logs', sess.graph)