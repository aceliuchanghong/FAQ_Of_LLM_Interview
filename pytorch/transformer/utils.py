import time
import math
import tiktoken
import torch
import torch.nn as nn
from pytorch.transformer.config import parsers


def read_book(filename='hongLouMeng.txt', path=None):
    args = parsers()
    if not path:
        path = args.data_path
    if not path + filename:
        print("no such books:" + path + filename)
        return
    with open(path + filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# TOKEN化文字 此处使用openai的tiktoken
def get_token(text, encoding):
    """
    文字转TOKEN
    :param text:
    :param encoding:
    :return:
    """
    encoding = tiktoken.get_encoding(encoding)
    tokenized_text = encoding.encode(text)
    # token先转化为张量
    tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)
    # 文本中最大的token的索引
    max_token_value = tokenized_text.max().item()
    return tokenized_text, max_token_value


def split_train_and_valid(token, percent=0.9):
    """
    获取测试和验证数据集
    :param token:
    :param percent:
    :return:
    """
    train_idx = int(len(token) * percent)
    # 索引 0 到 train_idx-1 的元素
    train_data = token[:train_idx]
    valid_data = token[train_idx:]
    return train_data, valid_data


def prepare_training_batch(train_data):
    """
    x_batch, y_batch 对应有 batch_size个批次,每个批次有context_length个长度
    :param train_data:
    :return:
    """
    args = parsers()
    data = train_data

    # tensor([61887, 64152, 29136, 48336])
    # 确保idxs中没有重复的索引
    idxs = torch.unique(torch.randint(low=0, high=len(data) - args.context_length - 1, size=(args.batch_size,)))
    while len(idxs) < args.batch_size:
        new_idx = torch.randint(low=0, high=len(data) - args.context_length - 1, size=(1,))
        idxs = torch.unique(torch.cat([idxs, new_idx]))

    x_batch = torch.stack([data[idx:idx + args.context_length] for idx in idxs])
    y_batch = torch.stack([data[idx + 1:idx + args.context_length + 1] for idx in idxs])
    """
    print('batch_size, sequence_length:', y_batch.size(), '\n',
          x_batch[0], '\n',
          'x_batch:', tiktoken.get_encoding(args.encoding).decode(x_batch[0].numpy()), '\n',
          'y_batch:', tiktoken.get_encoding(args.encoding).decode(y_batch[0].numpy()), '\n',
          len(idxs))
    """
    return x_batch, y_batch


def getInputEmbedding(max_token_value, x_batch, y_batch):
    """
    向量化输入端数据
    :param max_token_value:
    :param x_batch:
    :param y_batch:
    :return:
    """
    args = parsers()
    input_embedding_lookup_table = nn.Embedding(max_token_value + 1, args.d_model)
    x_batch_embedding = input_embedding_lookup_table(x_batch)
    y_batch_embedding = input_embedding_lookup_table(y_batch)
    """
    print("batch_size, sequence_length, d_model:", x_batch_embedding.shape, '\n',
          'x_batch_embedding[0].shape:', x_batch_embedding[0].shape)
    """
    return x_batch_embedding, y_batch_embedding


def addPositionalEncoding(x_batch_embedding, y_batch_embedding):
    """
    增加位置编码
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    :param x_batch_embedding:
    :param y_batch_embedding:
    :return:
    """
    args = parsers()

    # 一个全零的张量作为位置编码的查找表，形状为 (context_length, d_model)
    position_encoding_lookup_table = torch.zeros(args.context_length, args.d_model)

    # 使用 PyTorch 的 torch.arange() 函数生成一个从 0 到 args.context_length - 1 的浮点数序列
    # eg:tensor([0., 1., 2., 3., 4.])
    # 然后 .unsqueeze(1): 这个方法是在张量的维度上增加一个新的维度。在这里，我们在维度 1 上增加了一个新的维度
    # 最后生成了一个形状为 (context_length, 1) 的张量 position,其中包含了从 0 到 context_length - 1 的序列,表示了序列中每个位置的索引
    position = torch.arange(0, args.context_length, dtype=torch.float).unsqueeze(1)
    # 一个从 0 开始、步长为 2、不超过 args.d_model 的整数序列
    div_term = torch.exp(torch.arange(0, args.d_model, 2).float() * (- math.log(10000.0) / args.d_model))

    # 这一行代码计算了位置编码中偶数索引位置的值
    # [:, 0::2] 是 Python 中的切片语法，表示对所有行，从第 0 列开始，每隔 2 列进行操作
    position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
    # 这一行代码计算了位置编码中奇数索引位置的值
    # 表示对所有行，从第 1 列开始，每隔 2 列进行操作
    position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)

    # 将查找表的维度从 (context_length, d_model) 扩展为 (batch_size, context_length, d_model)
    position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(args.batch_size, -1, -1)
    x = x_batch_embedding + position_encoding_lookup_table
    y = y_batch_embedding + position_encoding_lookup_table
    # 每一句话的每一个单词的不同维度的权重
    # import pandas as pd
    # print(pd.DataFrame(x[0].detach().numpy()))
    return x, y


if __name__ == '__main__':
    start = time.time()
    args = parsers()
    # 读取文件
    text = read_book("hongLouMeng.txt", path="../data/books/")
    # 变成token
    token, max_token_value = get_token(text, args.encoding)
    # 获取训练的数据
    train_data, valid_data = split_train_and_valid(token)
    # 获取训练批次张量数据
    x_batch, y_batch = prepare_training_batch(train_data)
    # embedding
    x_batch_embedding, y_batch_embedding = getInputEmbedding(max_token_value, x_batch, y_batch)
    # 增加位置信息
    x, y = addPositionalEncoding(x_batch_embedding, y_batch_embedding)
    end = time.time()
    print(f"运行时间：{(end - start) / 60 % 60:.4f}分({end - start:.4f}秒)")
