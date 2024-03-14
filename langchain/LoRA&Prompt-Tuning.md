LoRA（Low-Rank Adaptation)

![img_2.png](..%2Fusing_files%2Fimg%2Flora%2Fimg_2.png)

见上图:LoRA 固定模型参数(W0)不变,将小的可训练适配器层(ΔW = BA)附加到模型上,并且只训练适配器

使用矩阵分解(奇异值分解或特征值分解)将语言模型的参数矩阵分解为较低秩的近似矩阵