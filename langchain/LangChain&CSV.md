### 基于存储在 CSV 文件中的数据构建 Q&A 系统
与使用 SQL数据库一样,使用 CSV文件的关键是提供对用于查询数据和与数据交互的工具LLM的访问权限
```text
1.将 CSV 加载到 SQL 数据库中，并使用 SQL 例子文档中写的方法
2.提供对 Python 环境LLM的访问权限，在该环境中，它可以使用 Pandas 等库与数据进行交互。
```
- 使用SQL与CSV数据进行交互，因为与任意Python相比，限制权限和清理查询更容易
```python
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import pandas as pd

df = pd.read_csv("titanic.csv")
engine = create_engine("sqlite:///titanic.db")
# 将 CSV 文件作为 SQLite 表加载
df.to_sql("titanic", engine, index=False)
```
- 创建一个 SQL 代理来与之交互
```python
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
agent_executor.invoke({"input": "what's the average age of survivors"})
```

[demo03.ipynb](LangChain%2Fdemo03.ipynb)

### Reference(参考文档)

* [LangChain-CSV](https://python.langchain.com/docs/use_cases/sql/quickstart)



