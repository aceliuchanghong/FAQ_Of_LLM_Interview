```text
1. `model_name`: 模型名称，默认为"./chatglm3-6b-base"，是您想要训练的模型的Huggingface名称。
2. `data_path`: 数据路径，默认为"formatted_samples.json"，指向训练数据的路径。
3. `output_dir`: 输出目录，默认为'./trained_model'，用于存储日志和检查点文件。
4. `training_recipe`: 训练配方，默认为"lora"，指定使用的训练方式（Lora训练或完整训练）。
5. `optim`: 优化器，默认为'paged_adamw_8bit'，指定要使用的优化器。
6. `batch_size`: 批量大小，默认为4，每个GPU的训练批量大小。增加以提高速度。
7. `gradient_accumulation_steps`: 梯度累积步数，默认为1，执行优化器步骤之前要累积的梯度数量。
8. `n_epochs`: 训练轮数，默认为5，要执行的优化器更新步数。
9. `weight_decay`: 权重衰减，默认为0.0，AdamW的L2权重衰减率。
10. `learning_rate`: 学习率，默认为1e-4。
11. `max_grad_norm`: 梯度裁剪最大范数，默认为0.3。
12. `gradient_checkpointing`: 是否使用梯度检查点，默认为True。
13. `do_train`: 是否进行训练，默认为True。
14. `lr_scheduler_type`: 学习率调度类型，默认为'cosine'。
15. `warmup_ratio`: 预热比例，默认为0.03。
16. `logging_steps`: 记录损失的频率，默认为1。
17. `group_by_length`: 将序列按相同长度分组为批次，默认为True。
18. `save_strategy`: 保存检查点的策略，默认为'epoch'。
19. `save_total_limit`: 最多保存检查点的数量，默认为3。
20. `fp16`: 是否使用fp16混合精度训练，默认为False。
21. `tokenizer_type`: 分词器类型，默认为"llama"。
22. `trust_remote_code`: 是否信任远程代码，默认为True。
23. `compute_dtype`: 模型的计算数据类型，默认为torch.float16。
24. `max_tokens`: 最大标记数，默认为4096。
25. `do_eval`: 是否进行评估，默认为True。
26. `evaluation_strategy`: 评估策略，默认为'epoch'。
27. `use_auth_token`: 是否使用auth token，默认为False。
28. `use_fast`: 是否使用快速分词器，默认为False。
29. `bits`: 量化模型的位数，默认为4。
30. `double_quant`: 是否通过双重量化压缩量化统计数据，默认为True。
31. `quant_type`: 量化数据类型，默认为"nf4"。
32. `lora_r`: Lora R维度，默认为64。
33. `lora_alpha`: Lora alpha值，默认为16。
34. `lora_dropout`: Lora dropout值，默认为0.05。
```


```text
1. `batch_size`: 指定每个GPU的训练批量大小。增加批量大小可以提高训练速度，但可能会导致内存不足或性能下降。
2. `gradient_accumulation_steps`: 指定累积梯度更新的步数。这可以帮助在内存受限的情况下训练使用更大批量大小的模型。
3. `weight_decay`: AdamW优化器的L2权重衰减率。它有助于防止模型过拟合，通过惩罚较大的权重值来提高泛化能力。
4. `max_grad_norm`: 梯度裁剪的最大范数。用于防止梯度爆炸问题，限制梯度的大小。
5. `lr_scheduler_type`: 学习率调度类型，控制学习率如何随时间变化。常见的类型包括常数学习率、余弦退火等。
6. `warmup_ratio`: 学习率预热比例，指定训练开始阶段用于逐渐增加学习率的比例。
7. `logging_steps`: 记录损失的频率，指定多少步之后记录一次训练损失。
8. `group_by_length`: 是否将序列按相同长度分组为批次，有助于节省内存并加快训练速度。
9. `save_strategy`: 检查点保存策略，指定何时保存检查点，可以是每个epoch或每个步骤。
10. `fp16`: 是否使用混合精度训练，通过减少计算精度来加快训练速度。
11. `max_tokens`: 指定每个训练示例的最大标记数，用于控制输入数据的大小。
12. `evaluation_strategy`: 评估策略，指定何时进行评估，可以是每个epoch或每个步骤。
13. `learning_rate`（学习率）是一个非常重要的超参数，用于控制模型在每次迭代中更新权重的速度。较大的学习率可能会导致模型在训练过程中跳过最优解，而较小的学习率可能会导致训练时间过长或者陷入局部最优解。通常，学习率是需要进行调整和优化的一个关键参数。
14. `quant_type`（量化数据类型）是用于指定模型量化过程中所使用的数据类型。在深度学习中，量化是一种技术，用于减少模型的存储空间和计算成本。通过将模型参数和激活值转换为较低位宽的数据类型（如8位整数），可以减少模型的内存占用和加速推理过程。`quant_type`参数允许您指定量化过程中所使用的具体数据类型，例如"fp4"表示4位浮点数，"nf4"表示4位无符号整数等。选择合适的量化数据类型对模型的性能和精度都有影响。
```
