# transformer类
import time
import math
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch.transformer.config import parsers

arg = parsers()


class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = arg.d_model
        self.dropout = arg.dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(arg.dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(arg.d_model, arg.d_model)
        self.Wk = nn.Linear(arg.d_model, arg.d_model)
        self.Wv = nn.Linear(arg.d_model, arg.d_model)
        # 来一个mask
        # self.register_buffer 用于向 PyTorch 模型注册一个持久化的参数，
        # 该参数在模型的训练过程中不会被更新，但可以被模型使用。在这里，它用于创建一个名为 mask 的矩阵，并将其注册为模型的缓冲区。
        self.register_buffer('mask', torch.tril(torch.ones(arg.context_length, arg.context_length)))

    def forward(self, x):
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # //是整数除法运算符，它会返回不大于结果的最大整数 这儿除num_heads是因为多头其实就是维度切分
        attention = Q @ K.transpose(-2, -1) / math.sqrt(arg.d_model // arg.num_heads)

        attention = attention.masked_fill(self.mask == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = attention @ V
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention() for _ in range(arg.num_heads)])
        self.projection_layer = nn.Linear(arg.d_model, arg.d_model)
        self.dropout = nn.Dropout(arg.dropout)

    def forward(self, x):
        self.heads = [head(x) for head in self.heads]
        out = torch.cat(self.heads, dim=-1)
        out = self.projection_layer(out)
        out = self.dropout(out)

        return out
