### Paddle(飞桨)

是一个由百度开发的深度学习框架,常用api：

1. 加载数据

```python
import paddle
from paddle.io import DataLoader
from paddle.vision.datasets import MNIST

train_dataset = MNIST(mode='train')
# 每个训练批次中包含64个样本 
# 在每个epoch开始时，会对训练数据进行随机洗牌，即打乱数据的顺序。这样做有助于增加训练的随机性，防止模型过度拟合训练数据的顺序
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

2. 构建模型

```python
import paddle.nn as nn

class MyModel(nn.Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        # 模型中添加一个全连接层（nn.Linear），输入维度为784，输出维度为10。
        self.fc = nn.Linear(784, 10)
    # 定义模型的前向传播方法，接收输入x并返回模型的输出
    def forward(self, x):
        x = self.fc(x)
        return x
```

3. 定义损失函数和优化器

```python
import paddle.optimizer as optim

model = MyModel()
# 定义交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义优化器（Adam优化器）
optimizer = optim.Adam(parameters=model.parameters(), learning_rate=0.001)
```

4. 训练模型

```python
for data in train_loader:
    images, labels = data
    output = model(images)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
```

### Paddle(飞桨)

是一个Google开发的深度学习框架，常用api：

1. 加载数据

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

2. 构建模型

```python
# 使用tf.keras.Sequential创建一个序贯模型，即按顺序堆叠各层的模型。
model = tf.keras.Sequential([
    # 添加一个Flatten层，用于将输入展平为一维向量，输入形状为(28, 28)。
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # 添加一个全连接层，包含128个神经元，使用ReLU激活函数。
    tf.keras.layers.Dense(128, activation='relu'),
    # 添加输出层，包含10个神经元，用于输出模型的预测结果
    tf.keras.layers.Dense(10)
])
```

3. 定义损失函数和优化器

```python
# 定义交叉熵损失函数
# 交叉熵损失函数通常用于多分类问题，计算模型输出与真实标签之间的差异。
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 定义优化器（Adam优化器）
# 使用Adam优化器初始化了一个优化器对象。在这里，learning_rate=0.001指定了学习率为0.001，
optimizer = tf.keras.optimizers.Adam()
```

4. 训练模型

```python
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
```

