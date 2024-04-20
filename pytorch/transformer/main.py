from pytorch.transformer.model import *
from pytorch.transformer.utils import *

if __name__ == '__main__':
    args = parsers()
    # 读取文件
    text = readBook()
    # 变成token
    token, max_token_value = getToken(text)
    # 获取训练的数据
    train_data, valid_data = split_train_and_valid(token)
    # 获取训练批次张量数据
    x_batch, y_batch = prepare_training_batch(train_data)
    # embedding
    x_batch_embedding, y_batch_embedding = getInputEmbedding(max_token_value, x_batch, y_batch)
    # 增加位置信息

    start = time.time()
    x, y = addPositionalEncoding(x_batch_embedding, y_batch_embedding)
    ffn = FeedForwardNetwork()
    ffn.forward(x)
