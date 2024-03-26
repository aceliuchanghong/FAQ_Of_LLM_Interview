### What it is
Transformer 是一种神经网络结构，它利用了自注意力（self-attention）机制和多层编码器（encoder）与解码器（decoder）层
从而有效地处理长距离依赖关系和捕获不同层次的文本信息。

步骤
```text
Input Embeddings
Positional Encodings
Layer Normalization
Feed Forward
Multi-Head Attention
Residual Connection
Encoder
Decoder
Linear Layer
Transformer
Task overview
Tokenizer
Dataset
Training loop
Validation loop
Attention visualization
```

Pytorch如何使用多gpu?
```text
# 将输入一个batch的数据均分成多份，分别送到对应的GPU进行计算，各个GPU得到的梯度累加
model= nn.DataParallel(model)
```
pytorch如何微调fine tuning?
```text
局部微调：加载了模型参数后，只想调节最后几层，其他层不训练，也就是不进行梯度计算
pytorch提供的requires_grad使得对训练的控制变得非常简单。

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# 替换最后的全连接层， 改为训练100类
# 新构造的模块的参数默认requires_grad为True
model.fc = nn.Linear(512, 100)
 
# 只优化最后的分类层
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)


全局微调：对全局微调时，只不过我们希望改换过的层和其他层的学习速率不一样，
这时候把其他层和新层在optimizer中单独赋予不同的学习速率。

ignored_params = list(map(id, model.fc.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     model.parameters())
 
optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': model.fc.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```
Pytorch如何实现大部分layer?
```text
pytorch可以实现大部分layer，这些层都继承于nn.Module
```
nn.Functional和nn.Module区别
```text
高层API方法：使用torch.nn.Module.****实现
低层API方法：使用低层函数方法，torch.nn.functional.****实现
```
反向传播的流程
```text
loss.backward()；反向传播 optimizer.step() 权重更新；optimizer.zero_grad() 导数清零
```












### Reference(参考文档)
[论文:Attention Is All You Need](..%2Fusing_files%2Fpaper%2Fpytorch2transformer.pdf)
* [论文](https://arxiv.org/abs/1706.03762)
* [PyTorch搭建Transformer视频](https://www.youtube.com/watch?v=ISNdQcPhsts&ab_channel=UmarJamil)
* [上述视频的文档github地址](https://github.com/aceliuchanghong/pytorch-transformer)
* [Pytorch面试题整理](https://blog.csdn.net/qq_43687860/article/details/132795944)
* [视觉算法工程师面经](https://blog.csdn.net/ly59782/article/details/120671350?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-9-120671350-blog-119749474.235^v43^control&spm=1001.2101.3001.4242.6&utm_relevant_index=12)
* [算法工程师面经](https://blog.csdn.net/julyedu_7/article/details/122473408?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-122473408-blog-119749474.235%5Ev43%5Econtrol&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-122473408-blog-119749474.235%5Ev43%5Econtrol&utm_relevant_index=2)
