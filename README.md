# NLP & KnowledgeGraph

------

> AttributeError: module 'tensorflow._api.v2.io.gfile' has no attribute 'get_filesystem'

解决办法:

> ```python
> import tensorflow as tf
> import tensorboard as tb
> tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
> ```

> ```python
> 保存模型
> # torch.save(rnn.state_dict(), 'rnn.pt')    保存
> m_state_dict = torch.load('linear.pt')
> new_m = Net()
> new_m.load_state_dict(m_state_dict)
> [w, b] = new_m.parameters()
> w = w.item()
> b = b.item()
> print(w, b)
> ```