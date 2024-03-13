### FAQ_Of_LLM_Interview

大模型算法岗面试题(含答案):常见问题和概念解析 "大模型面试题"、"算法岗面试"、"面试常见问题"
### 欢迎PR


### 暂时按照以下目录
```text
一、大模型算法基础
1. 自注意力机制(Self-Attention)
   1.1 自注意力机制的思路
   1.2 自注意力机制的计算过程
   1.3 自注意力机制的代码实现
2. 多头注意力机制(Multi-Head Attention)
   2.1 多头注意力机制的思路
   2.2 多头注意力机制的计算过程
   2.3 多头注意力机制的代码实现
3. 位置编码(Position Encoding)
   3.1 为什么需要位置编码
   3.2 位置编码的方法
      3.2.1 绝对位置编码
      3.2.2 相对位置编码
      3.2.3 旋转位置编码(RoPE)
4. 层归一化(Layer Normalization)
   4.1 层归一化的思路
   4.2 层归一化的计算过程
   4.3 层归一化的代码实现
5. 残差连接(Residual Connection)
   5.1 残差连接的作用
   5.2 残差连接的实现方式
6. Transformer 模型结构
   6.1 Transformer Encoder 结构
   6.2 Transformer Decoder 结构
   6.3 Transformer 模型整体结构

二、大模型算法优化
1. 注意力机制优化
   1.1 Linformer
   1.2 Longformer
   1.3 BigBird
   1.4 Reformer
2. 位置编码优化
   2.1 旋转位置编码(RoPE)
   2.2 ALiBi
3. 层归一化优化
   3.1 RMSNorm
   3.2 DeepNorm
4. 激活函数优化
   4.1 GELU
   4.2 SwiGLU
   4.3 ReGLU
5. 模型并行优化
   5.1 Tensor Parallelism
   5.2 Pipeline Parallelism
6. 模型压缩优化
   6.1 量化(Quantization)
   6.2 知识蒸馏(Knowledge Distillation)
   6.3 剪枝(Pruning)
   6.4 低秩分解(Low-Rank Decomposition)

三、大模型训练与微调
1. 大模型预训练
   1.1 预训练目标
   1.2 预训练数据
   1.3 预训练策略
2. 大模型微调
   2.1 全参数微调
   2.2 低秩微调(LoRA)
   2.3 Prompt Tuning
   2.4 Prefix Tuning
3. 多任务学习
   3.1 多任务学习的思路
   3.2 多任务学习的实现方式
4. 领域自适应
   4.1 数据收集与处理
   4.2 领域自适应方法
5. 评估指标
   5.1 自动评估指标
   5.2 人工评估指标

四、大模型推理与部署
1. 推理加速
   1.1 INT8量化
   1.2 FP16混合精度
   1.3 Tensor/Pipeline Parallelism
2. 模型裁剪
   2.1 前缀裁剪(Prefix Dropping)
   2.2 注意力头裁剪(Attention Head Pruning)
3. 上下文扩展
   3.1 流式处理(Streaming)
   3.2 重叠拼接(Overlapping)
4. 模型部署
   4.1 在线服务部署
   4.2 离线应用部署

五、大模型应用场景
1. 自然语言处理
   1.1 文本生成
   1.2 机器翻译
   1.3 问答系统
   1.4 文本摘要
2. 计算机视觉
   2.1 图像分类
   2.2 目标检测
   2.3 图像生成
3. 多模态
   3.1 视觉问答
   3.2 图文生成
   3.3 多模态检索
4. 其他领域
   4.1 蛋白质结构预测
   4.2 分子设计
   4.3 代码生成

六、大模型面临的挑战
1. 数据质量与隐私
2. 计算资源需求
3. 模型可解释性
4. 公平性与偏见
5. 安全性与可靠性
6. 能源消耗与环境影响
```
