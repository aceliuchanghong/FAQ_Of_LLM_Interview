# 辅助的工具函数
def readBook(filename='hongLouMeng.txt', path=None):
    """
    读取pytorch/data/books/下面的小说
    :param filename:
    :param path:
    :return:
    """
    if not path:
        path = 'pytorch/data/books/'
    if not path + filename:
        print("no such books:" + path + filename)
        return
    with open(path + filename, 'r', encoding='utf-8') as f:
        text = f.read()

    return text


# TOKEN化文字 此处使用openai的tiktoken
def getToken(text):
    """
    文字转TOKEN
    :param text:
    :return:
    """
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenized_text = encoding.encode(text)
    return tokenized_text


if __name__ == '__main__':
    text = readBook("sales_textbook.txt", path="../data/books/")
    token = getToken(text)
    print(len(token))

    print(token[:30])
