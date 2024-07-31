基本概念
```text
embedding: 
文本转向量

位置编码:
x = x_batch_embedding + position_encoding_lookup_table
y = y_batch_embedding + position_encoding_lookup_table

loss:
J(x) = (Wx + b - y)^2

param:
p = p - lr * J'(p)

layer norm:
数字缩放

残差连接:
数据+原始数据

softmax:
数字转百分比

ffn
前馈神经网络
```


超参数设置:

```
config.py
```

训练程序:

```
model.py
```

推理程序:

```
inference.py
```

工具类:

```
utils.py
```

### Reference(参考文档)

- [llm张代码](https://github.com/waylandzhang/Transformer-from-scratch)
