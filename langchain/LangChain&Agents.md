### Tools

```text
先简单说一下tools:tools是代理、链或LLM可用于与世界交互的接口

在构建自己的代理时，您需要向它提供可以使用的工具列表
组成:name description args_schema 
下面仅提供最简单的示例
```

```python
# 实例
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

tool.name
tool.description
tool.args  # {'query': {'title': 'Query', 'type': 'string'}}
tool.return_direct  # 该工具是否应该直接返回给用户
tool.run({"query": "langchain"})
tool.run("langchain")
```

定义一个tools

```python
# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

@tool
def search(query: str) -> str:
    """Look up things online."""
    return "LangChain"

print(search.name)
print(search.description)
print(search.args)
```

```text
# 输出
search
search(query: str) -> str - Look up things online.
{'query': {'title': 'Query', 'type': 'string'}}
```

另外一种tools定义
```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
```
### What's Agents

```text
主要包括:资源点==>retriever,大模型==>llm,工具(检索+其他)==>tools,提示词==>prompt,agent创建器

由于agents里面很多:self-determined, input-dependent sequence of steps 特性
所以需要LangSmith进行自动跟踪

export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="<your-api-key>"
export TAVILY_API_KEY="..."
```

```python
# 示例
# Initialize a toolkit
toolkit = ExampleTookit(...)
# Get list of tools
tools = toolkit.get_tools()
# Create agent
agent = create_agent_method(llm, tools, prompt)
```

Tools 示例

```python
# search 在线搜索
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults()
search.invoke("what is the weather in SF")

# Retriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

# tools
from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
tools = [search, retriever_tool]
```
开始创建Agents
```python
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# 准备一些必备的stuff
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 开始执行
agent_executor.invoke({"input": "hi!"})
agent_executor.invoke({"input": "how can langsmith help with testing?"})

# 添加历史
from langchain_core.messages import AIMessage, HumanMessage

agent_executor.invoke({"input": "hi! my name is bob", "chat_history": []})
agent_executor.invoke(
    {
        "chat_history": [
            HumanMessage(content="hi! my name is bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
        ],
        "input": "what's my name?",
    }
)
```
```text
prompt内容:
[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
 MessagesPlaceholder(variable_name='chat_history', optional=True),
 HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
 MessagesPlaceholder(variable_name='agent_scratchpad')]
```

### Reference(参考文档)

* [LangChain-Agents](https://python.langchain.com/docs/modules/agents/)
* [LangChain-tools](https://python.langchain.com/docs/modules/tools/custom_tools/)
