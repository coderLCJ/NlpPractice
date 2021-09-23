# NLP & KnowledgeGraph

------

> AttributeError: module 'tensorflow._api.v2.io.gfile' has no attribute 'get_filesystem'

解决办法:

> ```python
> import tensorflow as tf
> import tensorboard as tb
> tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
> ```