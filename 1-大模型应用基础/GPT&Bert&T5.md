## 经典语言模型:GPT、BERT、T5

### GPT

此处只写最简单介绍,要看代码实现移步`Transformer模型结构.md`

- 什么是GPT？

即 Generative Pre-trained Transformer（生成式预训练Transformer 模型）

- 算法简单理解？

1. G（生成式）

大家可以把简单把 AI 本身理解为我们应该都很熟悉的一次函数，只不过拥有很多参数：

y = (w1 * x1 + w2 * x2 + w3 * x3 + ……) + b

x 可以看出我们输入给 AI 的内容，w 我们已经得到的参数，b 是一个偏置值。

2. P（预训练）

就是上面 AI「学习」得到 w1、w2……和 b，也就是总结一般规律的过程。

3. T（变换器）

Transformer 是一种神经网络结构，它利用了自注意力（self-attention）机制和多层编码器（encoder）与解码器（decoder）层，从而有效地处理长距离依赖关系和捕获不同层次的文本信息。

Transformer 解决的问题，就是 AI 如何快速准确地理解上下文，并且以通用且优雅、简洁的方式。而「注意力机制」就是解决这个问题的关键。

自注意力机制：自注意力是一种计算文本中不同位置之间关系的方法。它为文本中的每个词分配一个权重，以确定该词与其他词之间的关联程度。通过这种方式，模型可以了解上下文信息，以便在处理一词多义和上下文推理问题时作出合适的决策。

跨注意力机制：跨注意力是一种计算两个不同文本序列中各个位置之间关系的方法。它为一个序列中的每个词分配权重，以确定该词与另一个序列中的词之间的关联程度。通过这种方式，模型可以捕捉到两个序列之间的相互关系，以便在处理多模态数据、文本对齐和多任务学习等问题时作出正确的决策。

### BERT

- 什么是BERT？

是 Google 在 2018 年提出的自然语言处理（NLP）模型 全称是 *Bidirectional Encoder Representations from Transformers*

- 算法简单理解？

1. 双向编码（Bidirectional Encoding）：

BERT是一个双向的语言表示模型，它能够同时考虑上下文信息，从而更好地理解句子的含义。
通过使用双向Transformer编码器，BERT能够在预测每个单词时同时考虑其左右上下文，从而更好地捕捉单词之间的关系。

2. BERT 的核心架构

BERT 的架构基于 Transformer 的 Encoder 部分。

```text
[Input Tokens] 
     ↓
[Token Embeddings + Segment + Position]
     ↓
[Transformer Encoder × N 层]
     ↓
[Pooled Output (CLS), Sequence Output]
```

3. 输入表示
BERT 的输入是一个 token 序列，经过三部分编码：
- **Token Embeddings**：将单词或子词（WordPiece）映射为向量。
- **Segment Embeddings**：区分不同句子（如句子 A 和 B）。
- **Position Embeddings**：表示 token 在序列中的位置。

这些嵌入相加后，输入到 Transformer 层。

```python
input_embedding = TokenEmbedding + SegmentEmbedding + PositionEmbedding
```

4. 补充 Segment 示例说明

BERT 原始模型只接受最多两个句子的组合 假设我们有以下输入：

```text
[CLS] I love NLP [SEP] It is amazing [SEP]
```

- `[CLS], I, love, NLP, [SEP]` 是句子 A → Segment ID = 0
- `It, is, amazing, [SEP]` 是句子 B → Segment ID = 1

那么 Segment Embedding 会是一个长度与输入序列一致的向量：

```python
segment_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1]
```

然后这个 `segment_ids` 会被查表得到对应的嵌入向量，并加到总输入中。

### T5


*Text-to-Text Transfer Transformer*

T5架构是Transformer的变种

### 总结：T5 vs. 标准Transformer的核心差别
| **方面**             | **标准Transformer**                     | **T5**                                     |
|----------------------|-----------------------------------------|--------------------------------------------|
| **任务范式**         | 序列到序列，任务特定                   | 文本到文本，统一框架                       |
| **位置编码**         | 绝对位置编码                           | 相对位置编码                               |
| **LayerNorm**        | post-LayerNorm                         | pre-LayerNorm                              |
| **训练方式**         | 监督训练                               | 预训练（span corruption）+微调             |
| **输出层**           | 生成序列，需任务特定头                 | 生成文本，通用输出                         |
| **灵活性**           | 任务特定，扩展性弱                     | 高度通用，适配多种任务                     |

**问题**：假设你要用T5和标准Transformer分别处理“将英文翻译成法语”和“情感分类”两个任务，两种模型分别需要做什么改动？
**例子**：
- 标准Transformer：翻译直接用，分类需加分类头。
- T5：翻译输入“translate to French:”，分类输入“sentiment:”，直接生成结果。

T5和YOLO的“通用框架”理念高度相似：
- **统一处理**：T5用文本到文本框架统一NLP任务，YOLO用单次前向传播统一目标检测。
- **预训练+微调**：两者通过大规模预训练降低用户门槛，微调适配具体任务。
- **用户友好**：用户只需提供任务特定数据（T5：带前缀的文本；YOLO：标注图像），无需改模型结构。

**T5的“YOLO”特性**：
- 架构（编码器+解码器）像YOLO的骨干+检测头，处理从输入到输出的完整流程。
- 任务前缀像YOLO的anchor box，引导模型聚焦特定任务。
- 预训练的span corruption像YOLO的COCO预训练，赋予模型通用知识。
- 