import torch

"""
True
2.2.2+cu121
"""
print(torch.cuda.is_available())
print(torch.__version__)
# 假设你有一个大小为 28x28 的灰度图像
image = torch.randn(28, 28)

# 但是 PyTorch 中要求通道维度在最前面，所以需要添加一个通道维度
# 使用 unsqueeze 在索引 0 的位置添加一个维度
image_with_channel = image.unsqueeze(0)

# 现在图像的维度是 (1, 28, 28)，表示有一个通道，高度和宽度分别为 28
print(image_with_channel.shape)
