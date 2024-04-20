### How

Q0:什么是tensor(张量)?

```text
Tensor实际上就是一个多维数组（multi-dimensional array）
```

![img.png](..%2Fusing_files%2Fimg%2FPyTorch2%2Fconcepts%2Fimg.png)

```python
import torch
import torch.nn as nn

x = torch.empty(5, 3)  # 其中的值未初始化
tensor([[-3.9719e-23, 1.3130e-42, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]])

x = torch.rand(5, 3)  # 其中的值是在区间 [0, 1) 内均匀分布的随机数。
tensor([[0.1721, 0.5466, 0.7132],
        [0.8289, 0.5445, 0.3396],
        [0.0855, 0.5145, 0.2988],
        [0.5537, 0.6823, 0.0242],
        [0.4761, 0.9997, 0.8869]])

zeros_tensor = torch.zeros(2, 3)  # 创建一个大小为 2x3 的张量，其中所有元素都初始化为 0
tensor([[0., 0., 0.],
        [0., 0., 0.]])

# 创建一个大小为 2x3 的张量，其中所有元素都初始化为 1
ones_tensor = torch.ones(2, 3)
tensor([[1., 1., 1.],
        [1., 1., 1.]])

# 创建一个大小为 2x3 的张量，其中所有元素都是从标准正态分布中随机抽样得到的
randn_tensor = torch.randn(2, 3)
tensor([[-1.6176, -0.3406, -0.1306],
        [-0.9964, 0.5069, 1.5283]])

# 直接根据数据创建张量
x = torch.tensor([5.5, 3])
tensor([5.5000, 3.0000])

# 基于现有的张量创建一个新的张量。如果用户没有提供新的值
x = x.new_ones(5, 3, dtype=torch.double)  # 创建了一个新的大小为 5x3 的张量 不指定任何参数时 元素会被初始化为 1
x = torch.randn_like(x, dtype=torch.float)
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[-1.2605, 0.6679, 1.0013],
        [-1.2571, -1.1844, -0.7229],
        [2.0382, 2.0700, 1.6898],
        [-0.6017, 1.5209, -1.3384],
        [0.5035, -0.4835, 2.1972]])

x = torch.eye(2,2) # 【单位矩阵 2*2】
tensor([[1., 0.],
        [0., 1.]])
```

Q1:pytorch 的 squeeze()? unsqueeze(),cat()

```python
# 扩展 unsqueeze() 方法可以在指定维度上增加一个维度，从而扩展张量的形状。
# 创建一个大小为 2 的张量
tensor = torch.tensor([1, 2])
# 在第一维度上增加一个维度
expanded_tensor = tensor.unsqueeze(0)

# squeeze() 用于删除张量中大小为 1 的维度。它的作用是压缩张量的维度，但并不改变张量中的元素数量
# 创建一个大小为 1x3x1x2 的张量
tensor = torch.randn(1, 3, 1, 2)
# 使用 squeeze() 方法压缩张量的维度
squeezed_tensor = tensor.squeeze()
print("原始张量形状:", tensor.shape)  # 输出: 原始张量形状: torch.Size([1, 3, 1, 2])
print("压缩后张量形状:", squeezed_tensor.shape)  # 输出: 压缩后张量形状: torch.Size([3, 2])

# 创建两个大小为 2x3 的张量
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
# 沿第一维度连接两个张量
concatenated_tensor = torch.cat((tensor1, tensor2), dim=0)
```

Q2:torch.nn.Conv2d?

```python
# torch.nn.Conv2d 是 PyTorch 中用于创建二维卷积层的类
# 例子:
torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
# in_channels：输入数据的通道数，如果输入数据是图像，则通道数为图像的颜色通道数（例如RGB图像为3）。
# out_channels：卷积核的数量，也是输出特征图的通道数。
# kernel_size：卷积核的大小，可以是整数或元组，如 (3, 3)。
# stride：卷积核在输入上移动的步长，默认为 1。
# padding：在输入的各个边界周围添加零值的层数。如果设置为 padding=(1, 2)，则在宽度方向上添加 1 层零值，高度方向上添加 2 层零值，默认为 0。
# dilation：控制卷积核的空洞卷积（dilated convolution）的空洞率，用于在卷积核元素之间插入间隔。默认为 1。
# groups：控制输入和输出之间的连接。当 groups 等于输入通道数和输出通道数时，每个输入通道都连接到输出通道。默认为 1。
# bias：是否添加偏置，默认为 True。
# padding_mode：当 padding 不为 0 时，指定如何填充。默认为 'zeros'，可以选择 'reflect' 或 'replicate'。

import torch
import torch.nn as nn

# 创建一个二维卷积层，输入通道数为3，输出通道数为64，卷积核大小为3x3，padding为1
conv_layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)
# 假设有一个输入张量 input_tensor，维度为 [batch_size, 3, height, width]
batch_size = 32
height = 128
width = 128
# 创建随机输入张量
input_tensor = torch.randn(batch_size, 3, height, width)
# 进行前向传播
output_tensor = conv_layer(input_tensor)
```

Q3:torch.Size,shape和dtype用法和解释,如何与numpy转换?

```shell
size 和 shape 都是用来描述张量（tensor）的维度的属性。它们提供了相同的信息，但是在不同的语境中使用。
size：size() 是 PyTorch 张量对象的一个方法，用于返回张量的维度。
这个方法返回的是一个包含每个维度大小的元组，元组的长度表示张量的维度。
例如，一个大小为 2x3 的张量，它的 size() 方法会返回 (2, 3)。

shape：shape 是一个属性，用于返回张量的维度。
它是一个直接获取张量形状的属性，而不是一个方法。
和 size() 方法一样，shape 也返回一个包含每个维度大小的元组。
例如，一个大小为 2x3 的张量，它的 shape 属性会返回 (2, 3)。

dtype:是用来描述张量中元素的数据类型的属性

# 创建一个大小为 2x3 的张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
# 使用 size() 方法获取张量的大小
size = tensor.size()
print("Size:", size)  # 输出: Size: torch.Size([2, 3])
# 使用 shape 属性获取张量的形状
shape = tensor.shape
print("Shape:", shape)  # 输出: Shape: torch.Size([2, 3])
# 创建一个大小为 2x3 的张量，并指定数据类型为双精度浮点数
tensor = torch.randn(2, 3, dtype=torch.double)


PyTorch 张量转换为 NumPy 数组
# 创建一个 PyTorch 张量
torch_tensor = torch.randn(3, 4)
# 将 PyTorch 张量转换为 NumPy 数组
numpy_array = torch_tensor.numpy()

NumPy 数组转换为 PyTorch 张量：
# 创建一个 NumPy 数组
numpy_array = np.random.randn(3, 4)
# 将 NumPy 数组转换为 PyTorch 张量
torch_tensor = torch.from_numpy(numpy_array)

注意:当将 NumPy 数组转换为 PyTorch 张量时，它们会共享相同的内存，因此修改其中一个会影响另一个。
当将 PyTorch 张量转换为 NumPy 数组时，并不会复制数据，而是共享相同的内存，
因此修改其中一个也会影响另一个。如果想要避免这种共享内存的情况，可以使用 .clone() 方法复制张量。
```

```python
from torch import tensor

x = tensor([[-1.2605, 0.6679, 1.0013],
            [-1.2571, -1.1844, -0.7229],
            [2.0382, 2.0700, 1.6898],
            [-0.6017, 1.5209, -1.3384],
            [0.5035, -0.4835, 2.1972]])

# torch.Size实际上是一个元组，所以它支持所有的元组操作
x.size()
torch.Size([5, 3])

y = torch.rand(5, 3)
x + y  # 或者 torch.add(x, y)
torch.add(x, y)

tensor([[-1.1266, 0.6920, 1.1550],
        [-0.3966, -0.6690, -0.1450],
        [2.2895, 2.8438, 2.5130],
        [0.0207, 1.6769, -1.1836],
        [0.9577, 0.0145, 2.3102]])

# 更多张量操作，包括转置（transposing）、索引（indexing）、切片（slicing）、
# 数学操作（mathematical operations）、线性代数（liner algebra）、随机数（random numbers）

# 创建一个大小为 3x3 的张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# 切片获取第一行的元素
first_row = tensor[0, :]
# 获取第一列的元素
first_column = tensor[:, 0]

# 创建一个大小为 2x3 的张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
# 重塑为大小为 3x2 的张量
reshaped_tensor = tensor.reshape(3, 2)
```

Q4:nn.Embedding的实现?

```python
# nn.Embedding 层的实现基于矩阵乘法，其内部维护一个由随机初始化的词向量组成的矩阵。
# 假设词汇表大小为10000，每个词语被表示为一个长度为100的向量
vocab_size = 10000
embedding_dim = 100
# 创建一个 Embedding 层
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
# 假设有一个输入张量 input_indices，包含了一批词语的索引
input_indices = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
# 将输入张量传递给 Embedding 层
embedded_vectors = embedding_layer(input_indices)
print(embedded_vectors.shape)  # 输出: torch.Size([2, 3, 100])

```

Q5:torch.optim.AdamW,SGD看看实现的原理,公式

```python
# Adam（Adaptive Moment Estimation）优化算法是一种结合了动量法（Momentum）和自适应学习率调整的优化算法。
# AdamW 是 Adam 优化算法的一种变体，其中 "W" 表示使用了权重衰减（Weight Decay）
# SGD（随机梯度下降）
```

![img_3.png](..%2Fusing_files%2Fimg%2FPyTorch2%2Fconcepts%2Fimg_3.png)

![img_2.png](..%2Fusing_files%2Fimg%2FPyTorch2%2Fconcepts%2Fimg_2.png)

![img_1.png](..%2Fusing_files%2Fimg%2FPyTorch2%2Fconcepts%2Fimg_1.png)


Q6:torch.argmax()

```shell
# torch.argmax() 是 PyTorch 中一个用于在指定维度上找到张量中最大值的索引的函数。
# 具体来说，torch.argmax() 返回沿着指定维度（dim 参数指定）上最大值的索引。
# 在这里，dim=1 表示在第一个维度（通常是列）上进行操作。
pred = [
    [0.1, 0.8, 0.1],   # 样本1的预测结果，最大值索引为1
    [0.3, 0.2, 0.5],   # 样本2的预测结果，最大值索引为2
    [0.5, 0.4, 0.1]    # 样本3的预测结果，最大值索引为0
]
torch.argmax(pred, dim=1)
# 将得到一个形状为 (3,) 的张量：
[1, 2, 0]
```

Q7:built_corpus里面经常有word_2_index = {"\<PAD>": 0, "\<UNK>": 1},何用? (这儿\是为了显示好看)

```text
word_2_index = {"<PAD>": 0, "<UNK>": 1}
填充和未知单词： 在处理文本序列时，通常会遇到长度不一致的情况。
为了处理这种情况，常常会使用一个特殊的标记来表示填充（padding）以及一个特殊的标记来表示未知单词（unknown word）。
这样可以保持所有文本序列的长度一致，并且能够处理模型未见过的单词。
```

Q8:Trainer的基础使用及原理

Q9:基础组件之Model

Q10:基础组件之Tokenizer

Q11:基础组件之Datasets

Q12:基础组件之Pipeline

Q13:基础组件之Evalaute

Q14:

Q15:

Q16:

Q17:

Q18:

Q19:

Pytorch如何使用多gpu?
```text
# 将输入一个batch的数据均分成多份，分别送到对应的GPU进行计算，各个GPU得到的梯度累加
dp:model= nn.DataParallel(model)
或者ddp
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


```text
PyTorch 是一个开源的机器学习库，它提供了丰富的工具和接口来构建和训练深度学习模型。下面是 PyTorch 的一些基本用法总结：

张量创建和操作:

使用 torch.tensor 创建张量:

import torch
x = torch.tensor([1.0, 2.0, 3.0])
张量属性和操作:

x.size()  # 获取张量的形状
x.shape   # 也可以使用 shape 属性
x.dtype   # 获取张量的数据类型
张量运算:

支持常见的数学运算:

y = torch.tensor([4.0, 5.0, 6.0])
z = x + y  # 张量相加
广播机制:

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([1.0, 2.0])
c = a + b  # 使用广播机制
自动微分:

使用 torch.autograd 追踪张量的计算历史:
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x**2
y.backward(torch.ones_like(x))  # 对 y 求导
模型构建和训练:

定义神经网络模型:

import torch.nn as nn
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)
训练模型:

model = SimpleNet()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
这些是 PyTorch 的一些基本用法，涵盖了张量操作、自动微分和模型构建训练等方面。
```

简单做一个交易分类

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. 数据预处理
# 有了处理好的数据集,包含特征和标签
"""
获取历史交易数据,包括交易金额、时间、地点、账户信息等特征。
对数据进行清洗,处理缺失值、异常值等。
将类别标签(正常、可疑、洗钱)编码为数值,如0、1、2。
"""


# 自定义数据集
class TransactionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 2. 构建模型
# 定义神经网络模型
class TransactionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TransactionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 3. 训练模型
# 准备数据
train_dataset = TransactionDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
input_size = train_features.shape[1]
hidden_size = 64
num_classes = 3  # 正常、可疑、洗钱
model = TransactionClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 4. 评估模型
# 准备测试数据
test_dataset = TransactionDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')

# 5. 模型部署
# 导出模型权重
torch.save(model.state_dict(), 'transaction_classifier.pth')

# 推理代码
new_transaction = ...  # 新的交易数据
model.load_state_dict(torch.load('transaction_classifier.pth'))
model.eval()
with torch.no_grad():
    output = model(new_transaction)
    _, predicted = torch.max(output.data, 1)
    print(f'Predicted class: {predicted.item()}')  # 0: 正常, 1: 可疑, 2: 洗钱
```

### Reference(参考文档)

* [Pytorch学习笔记总结](https://developer.aliyun.com/article/1055062)
* [学习的github](https://github.com/aceliuchanghong/Pytorch-NLP)
* [机器学习-有点广但不精](https://mofanpy.com/)
* [NLP-Tutorials](https://github.com/aceliuchanghong/NLP-Tutorials)
