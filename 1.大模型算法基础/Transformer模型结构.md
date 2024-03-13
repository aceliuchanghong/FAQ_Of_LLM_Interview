1. transformer模型结构图

![d4e71cf9dbc29549dc93408bc7a4fcf.png](..%2Fusing_files%2Fimg%2Ftransformer%2Fd4e71cf9dbc29549dc93408bc7a4fcf.png)

```text
transformer的左半部分是encoder（编码器），右半部分是decoder（解码器）

1.encoder是由6个相同的层堆叠构成的，即上图中N=6
每一层都有两个子层：
第一个子层是多头自注意力机制（multi-head self-attention mechanism）
第二个子层是全连接的前馈网络（feed forward network）
两个子层之间通过残差连接（residual connection），然后进行层归一化（layer normalization）

2.decoder也是由6个相同的层堆叠构成，即上图中N=6。
每一层都有三个子层：
第一层是带掩码的多头自注意力机制（Masked multi-head self-attention mechanism）
第二层是全连接的前馈网络（feed forward network）
第三层是多头注意力机制，类似于在encoder的第一子层中实现的机制。
在decoder，这种多头注意力机制接收来自前一个decoder层的查询，以及来自encoder输出的键和值。这允许decoder处理输入序列中的所有单词
```

2. Transformer为何使用多头注意力机制

```text
Multi-Head Attention
1.多头注意力机制允许模型同时关注不同的位置和语义信息,如果只使用一个头，模型可能会错过某些重要的依赖关系
2.多头可以使参数矩阵形成多个子空间，矩阵整体的size不变，只是改变了每个head对应的维度大小
3.可以充分利用现代硬件并行计算的能力
```

3. Transformer为什么Q和K使用不同的权重矩阵生成

```text
Transformer 模型的自注意力机制中，Q（查询矩阵）、K（键矩阵）和 V（值矩阵）
实际上是为了对X进行线性变换，为了提升模型拟合能力
def forward(self, q, k, v, mask=None):
```

![img.png](..%2Fusing_files%2Fimg%2Ftransformer%2Fimg.png)

4. 为什么要位置编码

```text
1.transformer 其他结构没有考虑到单词之间的顺序信息，而单词的顺序信息对于语义是非常重要的
2.位置编码向量与输入embedding具有相同的维度（因此可以相加），并且使用正弦和余弦函数
```

5. 为什么需要残差连接(Residual Connection)和层归一化(Layer Normalization)

```text
残差连接：解决神经网络的“退化现象”(过拟合)
层归一化：使得训练的模型更稳定，并且起到加快收敛速度的作用
```

6. 为什么要用带掩码masked？
```text
因为在机器翻译中，是一个先后顺序的过程，即翻译第i个单词时，我们只能看到第i-1
及其之前的单词。通过 Masked 操作可以防止知道i+1 个及以后的单词
```

7. Transformer架构与主导的 sequence transduction models有哪些不同之处？
```text
传统的 sequence transduction models通常基于复杂的循环神经网络（RNN）或卷积神经网络（CNN）
而Transformer完全依赖于注意力机制，摒弃了循环和卷积的使用。
```




### Reference(参考文档)

* [Transformer解析1](https://blog.csdn.net/weixin_45965387/article/details/130470040)
* [Transformer解析2](https://zhuanlan.zhihu.com/p/657268039)
* [面试总结](https://docs.google.com/document/d/1LP4eZdxo_ovhB6CnfFqi8Ufys1MqEh9ubxDyeNk58hw/edit)



