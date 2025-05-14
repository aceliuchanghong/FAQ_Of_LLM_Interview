### Tensorflow

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

水太深,讲不透,此处抛砖引玉

早前的时候用tf这个库多一点,当然现在做什么随机森林,梯度提升之类的分类,回归任务也有,建议tf这个遇到场景再看,方面有点多
