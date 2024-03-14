### Fine-tune和prompt-tune。
1. full fine-tune，也叫全参微调，bert微调模型一直用的这种方法，全部参数权重参与更新以适配领域数据，效果好。

2. prompt-tune, 包括p-tuning、lora、prompt-tuning、adaLoRA等delta tuning方法，
```
部分模型参数参与微调，训练快，显存占用少，效果可能跟FT（fine-tune）比会稍有效果损失，但一般效果能打平。
```

### What is LoRA
LoRA（Low-Rank Adaptation)

![img_2.png](..%2Fusing_files%2Fimg%2Flora%2Fimg_2.png)

见上图:LoRA 固定模型参数(W0)不变,将小的可训练适配器层(ΔW = BA)附加到模型上,并且只训练适配器

### How LoRA
使用矩阵分解(奇异值分解或特征值分解)将语言模型的参数矩阵分解为较低秩的近似矩阵

在参数矩阵分解之后，我们可以选择保留较低秩的近似矩阵


### Adapter Tuning 

![img.png](..%2Fusing_files%2Fimg%2Flora%2Fimg.png)


如上图,将其嵌入 Transformer 的结构里面，
在训练时，固定住原来预训练模型的参数不变，只对新增的 Adapter 结构进行微调

### P-Tuning



### Reference(参考文档)

* [Tune简单理解](https://zhuanlan.zhihu.com/p/660721012)

