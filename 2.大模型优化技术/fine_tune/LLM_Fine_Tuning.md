### What
```text
Mistral-7B Fine-Tuning: A Step-by-Step Log
```

Q1. 训练的模型如何合并呢?

Q2. 训练的模型如何保存呢?

Q3. loss为0(损失函数为0)如何解决?

Q4. 微调过程遇见了哪些问题?

### 模型转存到google drive云盘
```
# 克隆模型
!git lfs install
!git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 /content/sample_data/00

# 确保文件存在,搭载云盘
import os
my_files = '/content/drive/MyDrive/fine_tune_log'
for dirname, _, filenames in os.walk(my_files):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# 复制保存文件        
!cp -r /content/sample_data/00 /content/drive/MyDrive/fine_tune_log/files
```

### Reference(参考文档)
* [Mistral-7B实际微调](https://gathnex.medium.com/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8)
* [Github1页面](https://github.com/aceliuchanghong/chatglm3_6b_finetune)
* [Github2页面](https://github.com/aceliuchanghong/chatglm3-base-tuning)
* [讨论区](https://github.com/THUDM/ChatGLM3/discussions/253)
