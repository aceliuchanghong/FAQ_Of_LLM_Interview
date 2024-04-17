
### How

Q1:pytorch 的 squeeze? unsqueeze

Q2:torch.nn.Conv2d?

Q3:torch.Size和shape用法和解释

Q4:nn.Embedding用的什么模型?

Q5:torch.optim.AdamW,SGD看看实现的原理,公式

Q6:torch.argmax()
```python
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

Q7:built_corpus里面经常有word_2_index = {"\<PAD>": 0, "\<UNK>": 1},何用?
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
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

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
* [机器学习-很细](https://mofanpy.com/)
* [NLP-Tutorials](https://github.com/aceliuchanghong/NLP-Tutorials)
