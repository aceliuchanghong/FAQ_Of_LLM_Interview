做算力的公司

面试很专业
```text
1.微调细节,lora的参数,sft细节,p-tune,prompt tune(公式)之类的细节,(问的很多)
2.deepspeed之类的了解,然后这种加速框架细节
3.llama的ffn有几层,其layer Normal的区别于chatglm,glm的attention机制和其他有什么不同,他们和gpt底层是哪儿有差别
4.分布式训练dp和ddp有什么区别,底层,如何加速并行训练
5.cv之间的常用算法知道吗?
6.注意力的公式
7.rlhf之类的
8.项目实施的细节,每一个项目挨个问
9.langchain细节
10.残差连接,归一,公式和代码实现看看
11.位置编码,或者相对位置编码,公式以及什么场景,哪儿个模型用了
12.ffn的细节,公式之类的
13.多模态是什么?
还有很多我不知道的,我连关键字都记不住,QAQ,没办法我太菜了
```

### 微调理解
1. 微调参数理解:
```shell
python finetune.py \
    --dataset_path /data/nfs/guodong.li/data/alpaca_tokenize \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \   # 指定了每隔多少步保存一次模型检查点
    --save_total_limit 2 \  # 限制了要保留的最近检查点的总数
    --learning_rate 1e-4 \
    --fp16 \   # 使用16位浮点数(half-precision)进行训练,可以减少内存使用并加速训练,但可能会导致一些精度损失  传统上,大多数深度学习框架使用32位浮点数(FP32或single-precision)进行计算
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir /home/guodong.li/data/chatglm-6b-lora

args = {
    "data_name": "msra",  # 数据集名称
    "model_dir": "/root/autodl-tmp/chatglm-6b/",  # chatglm-6b地址，修改为自己的路径
    "lora_r": 8,  # lora_rank 决定了这个低秩矩阵的秩,秩越高,可以捕获更多的任务特定信息,但同时也会增加计算和存储开销。通常选择较小的秩
    "max_source_length": 128,  # instruct+query的最大长度
    "max_target_length": 32,  # answer的最大长度
    "instruct_column": "instruct",  # instruct列名
    "query_column": "query",  # query列名
    "response_column": "answer",  # answer列名
    "train_path": "data/msra/instruct_data/train.txt", # 训练数据，修改为自己数据
    "dev_path": "data/msra/instruct_data/dev.txt",  # 测试数据，修改为自己数据
    "ignore_pad_token_for_loss": True,  # 指示模型在计算损失时是否忽略填充标记
    "train_batch_size": 12,  # 训练batch_size
    "gradient_accumulation_steps": 1,  # 梯度累积的步数 梯度累积是一种用于在内存有限的情况下增加有效批次大小的技术。在每个训练步骤中,模型的梯度被计算和累积,但只有在累积达到指定的步数后才进行参数更新。这样,可以在内存不足以容纳大批次的情况下,模拟更大的批次大小
    "save_dir": "/root/autodl-tmp/msra_trainer/",  # 保存模型位置，修改为自己的路径
    "num_train_epochs": 1,  # 训练epoch
    "local_rank": -1,  # deepspeed所需，默认就好
    "log_steps": 10,  # 多少步打印一次结果
    "save_steps": 50,  # 多少步保存一次模型
    "deepspeed_json_path": "deepspeed.json" # deepspeed配置
}

peft_config:
  peft_type: LORA
  task_type: CAUSAL_LM  # 表示任务类型为因果语言建模(Causal Language Modeling)
  r: 8 # 这是LoRA中低秩矩阵的秩(rank)。秩决定了添加的低秩矩阵的大小和表达能力。较高的秩可以捕获更多的任务特定信息
  lora_alpha: 32 # 这是LoRA中的缩放因子(scaling factor)。缩放因子控制了低秩矩阵的初始值大小。较大的缩放因子可以加速收敛,但也可能导致不稳定性。通常选择适中的缩放因子(如16、32)可以在收敛速度和稳定性之间取得平衡。
  lora_dropout: 0.1 # LoRA中应用于低秩矩阵的dropout率。
```



### DeepSpeed
```text
DeepSpeed是微软开发的一个深度学习优化库,主要用于训练超大规模的深度学习模型

1.ZeRO (Zero Redundancy Optimizer):这是DeepSpeed的核心技术之一
通过优化模型参数在多个GPU之间的分布和更新方式,尽可能减少冗余,从而大幅降低了超大模型训练的显存占用,使得单机多卡可以训练更大的模型。
2.高效的大batch训练:DeepSpeed优化了训练循环,提供了高效的数据预取、转发、梯度累积等功能,可以支持非常大的batch size训练,显著提升训练吞吐。
```

12. ffn
```text
每一层经过attention之后，还会有一个FFN，
这个FFN的作用就是空间变换。FFN包含了2层linear transformation层，中间的激活函数是ReLu。
```
![img_10.png](..%2Fusing_files%2Fimg%2Ftransformer%2Fimg_10.png)

13. 多模态
```text
多模态指的是多种模态的信息，包括：文本、图像、视频、音频等
```

汗流浃背,还得多学习,多做做

### Reference

* [各种微调讲解](https://github.com/liucongg/ChatGLM-Finetuning)
* [chatglm的lora微调](https://zhuanlan.zhihu.com/p/621793987)
