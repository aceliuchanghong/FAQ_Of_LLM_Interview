## FAQ_Of_LLM_Interview

大模型算法岗面试题(含答案):
常见问题和概念解析 "大模型面试题"、"算法岗面试"、"面试常见问题"、"大模型算法面试"、"大模型应用基础"

- [面试必问问题](面试必问问题.md)

### 个人理解

#### 1. 数学与编程基础
- 线性代数：掌握矩阵运算、特征值与向量空间变换。
- 多元微积分及偏导数：梯度下降和反向传播原理，链式法则。
- 统计学与概率论：理解特征分布、贝叶斯推理及概率预测模型。
- PyTorch 或类似框架：熟悉模型构建逻辑、架构设计、损失函数定义及训练流程。通过实践编写简单网络，逐步过渡到复杂模型

#### 2. 模型架构
- Transformer 及其变体：注意力机制、位置编码和多头自注意力，重点在预训练与微调策略。
- 前馈神经网络（FFN）：作为非线性变换模块，提升模型表达能力。
- 混合专家模型（MoE）：探索稀疏激活机制。
- 扩散模型与多模态架构：噪声注入与去噪过程，以及跨模态融合技术。重点分析如何将文本条件融入图像生成，实现条件概率建模。
- 高效优化技术：参数高效微调（PEFT）、模型量化与知识蒸馏。

#### 3. 文档处理
- 检索增强生成（RAG）：向量数据库、嵌入模型和检索优化技巧。
- 知识图谱与多模态文档：处理 PDF、网页或图像，集成到大模型 pipeline 中。

#### 4. 强化学习
- 基础概念：理解 Markov 决策过程、价值函数与策略梯度。从 Q-Learning 到 DQN，掌握探索-利用权衡。
- 常用算法：PPO,DPO,GRPO...
- 大模型整合：探索 RL 与 Transformer 的结合,Agent 系统中使用奖励模型指导生成。
- 系列完整教程链接: [强化学习](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzU2NDAxMDMxOQ==&action=getalbum&album_id=3994192799708102658)

#### 5. Agent 及评测
- Agent 系统搭建 langgraph
- 追踪 langfuse
- 评测 重点是数据

#### 6. 英语及其他扩展
- 英语能力：需要多读,积累词汇和语感。

### 必备知识

在阅读本库前，建议补充一些数学相关知识
- [数学知识](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzU2NDAxMDMxOQ==&action=getalbum&album_id=3783886508759777283)


### 其他

> 吐槽: 大模型这方向真的卷,新paper,新模型,东西出的比我读的还快
>
> 因为作者当时对好些内容不太懂,所以写的目录好些根本就没写完,而现在又没有继续整理的打算了,之后这个库预期只更新面试记录了


### *Star History*
[![Star History Chart](https://star-history.dera.page/svg?repos=aceliuchanghong/FAQ_Of_LLM_Interview&type=Date)](https://star-history.dera.page/#aceliuchanghong/FAQ_Of_LLM_Interview&type=Date)
