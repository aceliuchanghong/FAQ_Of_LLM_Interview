硬件资源很足,主要做算力的

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

自己理解:
1.lora
```shell
python finetune.py \
    --dataset_path /data/nfs/guodong.li/data/alpaca_tokenize \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir /home/guodong.li/data/chatglm-6b-lora

args = {
    "data_name": "msra",  # 数据集名称
    "model_dir": "/root/autodl-tmp/chatglm-6b/",  # chatglm-6b地址，修改为自己的路径
    "lora_r": 8,  # lora参数 对应 lora_rank 决定了这个低秩矩阵的秩,秩越高,可以捕获更多的任务特定信息,但同时也会增加计算和存储开销。通常选择较小的秩
    "max_source_length": 128,  # instruct+query的最大长度
    "max_target_length": 32,  # answer的最大长度
    "instruct_column": "instruct",  # instruct列名
    "query_column": "query",  # query列名
    "response_column": "answer",  # answer列名
    "train_path": "data/msra/instruct_data/train.txt", # 训练数据，修改为自己数据
    "dev_path": "data/msra/instruct_data/dev.txt",  # 测试数据，修改为自己数据
    "ignore_pad_token_for_loss": True,  # 默认就好
    "train_batch_size": 12,  # 训练batch_size
    "gradient_accumulation_steps": 1,  # 默认就好
    "save_dir": "/root/autodl-tmp/msra_trainer/",  # 保存模型位置，修改为自己的路径
    "num_train_epochs": 1,  # 训练epoch
    "local_rank": -1,  # deepspeed所需，默认就好
    "log_steps": 10,  # 多少步打印一次结果
    "save_steps": 50,  # 多少步保存一次模型
    "deepspeed_json_path": "deepspeed.json" # deepspeed配置
}

peft_config:
  peft_type: LORA
  task_type: CAUSAL_LM
  r: 8
  lora_alpha: 32
  lora_dropout: 0.1
```







汗流浃背,还得多看看,多做做

### Reference

* [各种微调讲解](https://github.com/liucongg/ChatGLM-Finetuning)
* [chatglm的lora微调](https://zhuanlan.zhihu.com/p/621793987)
