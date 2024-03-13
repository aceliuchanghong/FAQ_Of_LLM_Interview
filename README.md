### FAQ_Of_LLM_Interview

大模型算法岗面试题(含答案):常见问题和概念解析 "大模型面试题"、"算法岗面试"、"面试常见问题"
### 欢迎PR


### 暂时按照以下目录
```text
FAQ_Of_LLM_Interview
|
├── LICENSE
├── README.md
├── prompt.md
├── 面试必问问题.md
├── 1.大模型算法基础/
│   ├── PyTorch从0开始编写Transformer.md
│   ├── PyTorch搭建神经网络.md
│   ├── Transformer模型结构.md
│   ├── 大模型的泛化能力.md
│   └── 预训练与微调.md
└── 2.大模型优化技术/
    ├── 推理优化.md
    ├── 模型压缩技术.md
    ├── 模型结构优化.md
    └── 训练优化.md
```
一、大模型基础知识
1. Transformer模型结构
   - Encoder-Decoder架构
   - Self-Attention机制
     - Scaled Dot-Product Attention
     - 计算复杂度分析
   - Multi-Head Attention
     - 并行计算与参数共享
     - 不同Head的作用与解释
   - Feed-Forward Network
     - 前馈神经网络结构
     - 激活函数选择(ReLU、GeLU等)
   - Residual Connection
     - 残差连接的作用与优势
     - Pre-Norm与Post-Norm的区别
   - Layer Normalization
     - 归一化的必要性与效果
     - Layer Norm与Batch Norm的区别
2. 预训练与微调
   - 无监督预训练(如BERT、GPT等)
     - Masked Language Modeling(MLM)
     - Next Sentence Prediction(NSP)
     - Permutation Language Modeling(PLM)
   - 有监督微调(如分类、序列标注等)
     - 微调方法(如全参数微调、Prompt Tuning等)
     - 微调过拟合问题与应对策略
   - 领域自适应
     - 领域数据集构建
     - 持续学习与增量学习
   - 多任务学习
     - 多任务学习的优势与挑战
     - 任务间知识迁移与干扰问题
3. 评估指标
   - 困惑度(Perplexity)
     - 定义与计算方法
     - 困惑度的局限性
   - BLEU、ROUGE等生成任务评估指标
     - 指标的定义与计算方法
     - 不同指标的适用场景与局限性
   - 精确率、召回率、F1等分类任务评估指标
     - 指标的定义与计算方法
     - 不平衡数据集下的评估问题
4. 注意力可视化与分析
   - 注意力矩阵的可视化方法
   - 注意力分布的统计分析
   - 注意力机制的可解释性
5. 大模型的参数量与计算量分析
   - 不同规模模型的参数量比较
   - FLOPs与推理速度估算
   - 模型压缩与加速方法
6. 大模型的泛化能力与鲁棒性
   - 零样本学习能力
   - 抗干扰能力与对抗攻击
   - 数据分布变化下的泛化能力
7. 大模型的知识表示与存储
   - 知识的隐式编码方式
   - 知识的显式存储与更新
   - 知识的提取与应用
8. PyTorch搭建神经网络


二、大模型优化技术
1. 模型结构优化
   - Attention机制变体(如Sparse Attention、Linear Attention等)
     - 稀疏注意力机制的原理与优势
     - 线性注意力机制的原理与优势
   - 位置编码优化(如相对位置编码)
     - 绝对位置编码的局限性
     - 相对位置编码的原理与优势
   - 激活函数选择(如GeLU、Swish等)
     - 不同激活函数的特点与适用场景
     - 激活函数对模型性能的影响
   - 归一化方式改进(如Pre-LN、AdaNorm等)
     - Pre-LN与Post-LN的区别与优劣
     - AdaNorm的原理与效果
2. 训练优化
   - 优化器选择(如AdamW、Lion等)
     - 不同优化器的原理与特点
     - 优化器超参数的选择与调优
   - 学习率调度(如Warmup、Cosine Annealing等)
     - 学习率调度的必要性
     - 不同调度策略的原理与效果
   - 梯度裁剪(Gradient Clipping)
     - 梯度裁剪的作用与实现方式
     - 梯度裁剪阈值的选择
   - 正则化技术(如Dropout、Weight Decay等)
     - 不同正则化技术的原理与作用
     - 正则化技术的适用场景与局限性
   - 数据增强(如词替换、回译等)
     - 数据增强的必要性
     - 不同数据增强方法的原理与效果
3. 推理优化
   - 模型量化(如FP16、INT8等)
     - 量化的原理与实现方式
     - 量化对模型性能与速度的影响
   - 模型裁剪(如Pruning、Knowledge Distillation等)
     - 剪枝的原理与实现方式
     - 知识蒸馏的原理与实现方式
   - 可扩展注意力(如Longformer、Big Bird等)
     - 可扩展注意力机制的原理
     - 不同可扩展注意力模型的特点与适用场景
   - 模型并行(如Tensor Parallelism、Pipeline Parallelism等)
     - 不同并行策略的原理与实现方式
     - 并行策略的优缺点与适用场景
   - 推理加速(如TensorRT、ONNX Runtime等)
     - 推理加速框架的原理与使用方法
     - 推理加速对模型性能与速度的影响
4. 模型压缩技术
   - 低秩分解(如SVD、Tucker分解等)
     - 低秩分解的原理与实现方式
     - 低秩分解对模型性能与速度的影响
   - 参数共享(如Cross-Layer Parameter Sharing等)
     - 参数共享的原理与实现方式
     - 参数共享对模型性能与速度的影响
   - 架构搜索(如NAS、AutoML等)
     - 架构搜索的原理与实现方式
     - 架构搜索的计算成本与效果评估

### 吐槽
```text
大模型这方向真的卷，新模型，新paper疯狂出，东西出的比我读的快
```