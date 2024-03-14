### LangChain

- Chains
```text
链是指调用序列 - 无论是对 LLM、 工具还是数据预处理步骤。支持的主要方法是使用 LCEL
1.create_retrieval_chain
2.create_sql_query_chain
```
- Agents
```python
from langchain import hub

# 系统的 prompt==>Agents (我理解)
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages
```
- Advanced Retrieval Strategies
```text
LangChain旨在帮助构建问答应用程序，以及更普遍的RAG应用程序
```

![langchain_stack.png](..%2Fusing_files%2Fimgs%2FLangChain%2Flangchain_stack.png)

### 典型的 RAG(Retrieval Augmented Generation 检索增强生成) 应用程序包括:

- Indexing(索引)

Load加载：首先我们需要加载数据。为此，我们将使用 DocumentLoaders。

Split拆分：文本拆分器将大 Documents 块分解为更小的块。这对于索引数据和将其传递到模型都很有用，因为大块更难搜索，并且不适合模型的有限上下文窗口。

Store存储：我们需要某个地方来存储和索引我们的拆分，以便以后可以搜索它们。这通常使用 VectorStore 和 Embeddings 模型来完成。

- Retrieval and generation(检索和生成)

Retrieve检索：给定用户输入，使用 Retriever 从存储中检索相关的拆分。

Generate生成：ChatModel / LLM 使用包含问题和检索到的数据的提示生成答案

## Example
[demo01.py](LangChain%2Fdemo01.py)
1. 加载 使用 WebBaseLoader，它用于 urllib 从 Web URL 加载 HTML 并将其 BeautifulSoup 解析为文本
```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
```
2. 拆分 文档长度超过 42k 个字符。这太长了，无法适应许多模型的上下文窗口 我们将文档拆分为 1000 个字符的块，块之间有 200 个字符重叠。重叠有助于减少将语句与与之相关的重要上下文分开的可能性

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
```
3. 存储  使用 Chroma 向量存储和 OpenAIEmbeddings 模型在单个命令中嵌入和存储所有文档拆分
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Embedding其实是将每个单词或其他类型的标记（如字符、句子或者文档）转换为一个固定长度的向量
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
```
4. 检索
```python
# LangChain 定义了一个 Retriever 接口，该接口包装了一个索引，该索引可以返回给定 Documents 的字符串查询相关。
# "k": 6 表示对于每个查询，检索器应该返回最相似的前6个结果
# retriever 是一个检索器
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# 执行
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
```
5. 生成
```python
# 使用一个llm模型
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# 使用一个 RAG 的提示，该提示已签入 LangChain 提示中心
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")
```
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# 使用 LCEL Runnable 协议来定义链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)
```
[demo01.ipynb](LangChain%2Fdemo01.ipynb)

### Reference(参考文档)

* [LangChain](https://python.langchain.com/docs/use_cases/question_answering/quickstart)

