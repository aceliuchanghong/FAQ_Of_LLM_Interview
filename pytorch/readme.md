### install

```
下载/官网
https://pytorch.org/

查看驱动版本
nvidia-smi
NVIDIA-SMI 546.12                 Driver Version: 546.12       CUDA Version: 12.3

开始安装(较大)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  --proxy=127.0.0.1:10809 
```

### What

- Q1:pytorch和tensorflow核心区别是什么?
- A1:动态图优先还是静态图优先


- Q2:动态图/静态图是什么?
- A2:静态图就像是一个很复杂的定义好的函数,执行中间是不可以修改的


- Q3:什么是深度学习?
- A3:是机器学习的分支，是一种以人工神经网络为架构，对资料进行表征学习的算法
  深度学习中的形容词“深度”是指在网络中使用多层,是机器学习中一种基于对数据进行表征学习的算法
  表征学习的目标是寻求更好的表示方法并建立更好的模型来从大规模未标记数据中学习这些表示方法

### 算法

#### 梯度下降算法(Gradient Descent Optimization)

```text
神经网络模型训练最常用的优化算法
用于更新模型参数以使损失函数最小化。
其核心思想是通过沿着损失函数的梯度方向迭代地更新参数，直至达到损失函数的局部最小值或全局最小值。

梯度下降算法的步骤如下：
1.初始化模型参数，如权重和偏置。
2.计算损失函数关于参数的梯度。
3.根据梯度的反方向更新参数，减小损失函数的值。
重复步骤2和步骤3，直到达到停止条件（如达到最大迭代次数、损失函数的变化小于某个阈值等）。
```

![img.png](..%2Fusing_files%2Fimg%2FPyTorch2%2Flinear%2Fimg.png)

损失函数公式

![img_1.png](..%2Fusing_files%2Fimg%2FPyTorch2%2Flinear%2Fimg_1.png)

![img_2.png](..%2Fusing_files%2Fimg%2FPyTorch2%2Flinear%2Fimg_2.png)

### Reference(参考文档)

* [常用的梯度下降算法](https://zhuanlan.zhihu.com/p/31630368)
* [pytorch教学视频](https://www.bilibili.com/video/BV1TN411k7hT)
* [配套代码ppt的github库](https://github.com/aceliuchanghong/Deep-Learning-with-PyTorch-Tutorials)
