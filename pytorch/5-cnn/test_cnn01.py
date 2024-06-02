import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # 定义第一个卷积层
        # 输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 定义第二个卷积层
        # 输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义第一个池化层
        # 使用2x2的窗口进行最大池化
        self.pool = nn.MaxPool2d(2, 2)
        # 定义第一个全连接层
        # 输入特征数为16*5*5，输出特征数为120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义第二个全连接层
        # 输入特征数为120，输出特征数为84
        self.fc2 = nn.Linear(120, 84)
        # 定义输出层
        # 输入特征数为84，输出特征数为10（假设有10个类别）
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 第一层卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 将多维张量展平为一维
        x = x.view(-1, 16 * 5 * 5)
        # 第一个全连接层 + ReLU
        x = F.relu(self.fc1(x))
        # 第二个全连接层 + ReLU
        x = F.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x


# 创建CNN实例
cnn = ConvolutionalNeuralNetwork()
print(cnn)
# input = torch.randn(batch_size, in_channels, width, heigth)
random_input = torch.randn(1, 1, 32, 32)
cnn(random_input)
