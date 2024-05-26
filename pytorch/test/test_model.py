import time

from pytorch.transformer.config import parsers
from pytorch.transformer.model import ScaledDotProductAttention
from pytorch.transformer.utils import read_book, get_token, split_train_and_valid, prepare_training_batch, \
    getInputEmbedding, addPositionalEncoding

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
    print(f"准备数据运行时间：{(end - start) / 60 % 60:.4f}分({end - start:.4f}秒)")

    start1 = time.time()
    # 创建ScaledDotProductAttention对象
    attention = ScaledDotProductAttention()
    output = attention(x)

    end1 = time.time()
    print(f"model运行时间：{(end1 - start1) / 60 % 60:.4f}分({end1 - start1:.4f}秒)")
