
### How

Q1:pytorch 的 squeeze? unsqueeze

Q2:torch.nn.Conv2d?

Q3:torch.Size和shape用法和解释

Q4:nn.Embedding用的什么模型?

Q5:torch.optim.AdamW,SGD看看实现的原理,公式

Q6:torch.argmax()
```python
# torch.argmax() 是 PyTorch 中一个用于在指定维度上找到张量中最大值的索引的函数。
# 具体来说，torch.argmax() 返回沿着指定维度（dim 参数指定）上最大值的索引。
# 在这里，dim=1 表示在第一个维度（通常是列）上进行操作。
pred = [
    [0.1, 0.8, 0.1],   # 样本1的预测结果，最大值索引为1
    [0.3, 0.2, 0.5],   # 样本2的预测结果，最大值索引为2
    [0.5, 0.4, 0.1]    # 样本3的预测结果，最大值索引为0
]
torch.argmax(pred, dim=1)
# 将得到一个形状为 (3,) 的张量：
[1, 2, 0]
```

Q7:built_corpus里面经常有word_2_index = {"\<PAD>": 0, "\<UNK>": 1},何用?
```text
word_2_index = {"<PAD>": 0, "<UNK>": 1}
填充和未知单词： 在处理文本序列时，通常会遇到长度不一致的情况。
为了处理这种情况，常常会使用一个特殊的标记来表示填充（padding）以及一个特殊的标记来表示未知单词（unknown word）。
这样可以保持所有文本序列的长度一致，并且能够处理模型未见过的单词。
```

Q8:

Q9:

Q10:

Q11:

Q12:

Q13:

Q14:

Q15:

Q16:

Q17:

Q18:

Q19:


























### Reference(参考文档)

* [Pytorch学习笔记总结](https://developer.aliyun.com/article/1055062)
* [学习的github](https://github.com/aceliuchanghong/Pytorch-NLP)
* [机器学习-很细](https://mofanpy.com/)
* [NLP-Tutorials](https://github.com/aceliuchanghong/NLP-Tutorials)
