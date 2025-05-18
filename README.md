## FAQ_Of_LLM_Interview

大模型算法岗面试题(含答案):
常见问题和概念解析 "大模型面试题"、"算法岗面试"、"面试常见问题"、"大模型算法面试"、"大模型应用基础"

- [面试必问问题](面试必问问题.md)

### Prompt--授人以鱼不如授人以渔
```
以老师教导学生的风格，教授我 [xxx]，循序渐进，言简意赅。  
1. 从日常生活中的实际场景引入，激发兴趣。  
2. 逐步引导，推导核心概念或公式，用通俗语言解释清楚。  
3. 深入讲解关键细节或进阶内容，保持清晰。[非常重要--深入讲解]  
4. 最后提供相关代码示例（如果适用），并简要解释代码逻辑。
确保每步逻辑连贯，语言亲切，像老师一样耐心引导。
```

### 个人理解

```
第一是对于模型的了解,大模型涉及很宽,需要花费很多时间学习,但是一通百通
    以下4个内容复习完再去看大模型就像张无忌学了九阳神功一样，可以修炼模型大挪移了：
    - 复习线性代数：理解矩阵与样本数据特征值间的变换关系。
    - 复习(多元)微积分及(偏)导数：理解神经网络反向更新。
    - 复习统计学：理解特征值分布与模型基于概率预测原理（llm、diffusion都是）。
    - 复习pytorch: 训练模型逻辑如何,架构如何,损失函数如何。

第二是对于文档的处理,不管什么项目都需要处理文档,非常重要
    可以看看IBM的RAG竞赛的冠军方案

(现在还需要对于强化学习了解才行..)
```

### 目录

```text
FAQ_Of_LLM_Interview/
|
├── 面试必问问题.md
├── 1-大模型应用基础/
│   ├── CNN卷积神经网络基础.md
│   ├── GPT&Bert&T5.md
│   ├── PyTorch从0开始编写Transformer.md
│   ├── PyTorch搭建神经网络.md
│   ├── Pytorch基本用法.md
│   ├── RNN循环神经网络基础.md
│   ├── Transformer模型结构.md
│   ├── Yolo基础知识.md
│   ├── paddle.md
│   ├── tensorflow.md
│   ├── 向量数据库.md
│   ├── 大模型的泛化能力.md
│   ├── 聚类.分类_算法.md
│   ├── 训练与推理.md
│   └── image/
│       └── GPT&Bert&T5/
├── 2-大模型优化技术/
│   ├── 常见大模型调用代码.md
│   ├── 微调优化.md
│   └── fine_tune/
│       ├── LLM_Fine_Tuning.ipynb
│       └── LLM_Fine_Tuning.md
├── 3-面试问题记录/
│   ├── 个人存储的各方向速成问题及解答-2025.md
│   ├── 2024_interview_log/
│   │   ├── ant.md
│   │   ├── atom.md
│   │   ├── liantong.md
│   │   ├── pdd.md
│   │   ├── relx.md
│   │   ├── saikai.md
│   │   ├── torch.md
│   │   ├── txyz.md
│   │   └── ucloud.md
│   └── 2025_interview_log/
│       └── 喆塔信息.md
├── 4-分布式训练篇/
│   ├── Accelerate-使用进阶.md
│   ├── DataParallel原理与应用.md
│   ├── DeepSpeed.md
│   ├── Distributed-DataParallel分布式数据并行原理与应用.md
│   └── 分布式训练与环境配置.md
├── 5-高效微调篇/
│   ├── LoRA..ETC.md
│   ├── Lora 原理与实战.md
│   ├── P-Tuning 原理与实战.md
│   ├── PEFT 进阶操作.md
│   ├── Prefix-Tuning 原理与实战.md
│   ├── Prompt-Tuning原理与实战.md
│   └── fine-tune参数解释.md
├── 6-强化学习基础/
│   └── readme.md
└── langchain/
    ├── LC&Extract.md
    ├── LangChain&Agents.md
    ├── LangChain&CSV.md
    ├── LangChain&LCEL.md
    ├── LangChain&SQL.md
    ├── LangChain&Server&Cli.md
    └── LangChain.md
```

### 必备知识

在阅读本文前，建议补充一些相关知识。若你之前未了解过相关原理，可以参考以下的链接：

* [姐控的机器学习记录](https://github.com/aceliuchanghong/my_lm_log)
* [[旧]姐控的大模型执行记录](https://github.com/aceliuchanghong/large_scale_models_learning_log)

### 特别鸣谢
- [张老师](https://github.com/zyxcambridge)(提供了此库的最开始的思路,没有他就没有此库)
- [赵老师](https://未提供链接,hh.com)(无偿回答了很多问题)
- [Kesoft](https://github.com/Kesoft)(添加了pr)

### 吐槽

```text
大模型这方向真的卷,新paper,新模型疯狂出,东西出的比我读的还快.
```

### 欢迎PR和联系我

![推广二维码](using_files/wechat/self_qr.png)

### *Star History*
[![Star History Chart](https://api.star-history.com/svg?repos=aceliuchanghong/FAQ_Of_LLM_Interview&type=Date)](https://www.star-history.com/#aceliuchanghong/FAQ_Of_LLM_Interview&Date)
