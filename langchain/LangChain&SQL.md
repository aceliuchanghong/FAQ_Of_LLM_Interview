### 在 SQL 数据库上创建 Q&A chain和 Agent 的基本方法
概括地说，任何 SQL 链和代理的步骤都是：
```text
1.将问题转换为 SQL 查询：模型将用户输入转换为 SQL 查询。
2.执行SQL查询：执行SQL查询。
3.回答问题：模型使用查询结果响应用户输入
```
![img.png](..%2Fusing_files%2Fimgs%2FLangChain%2Fimg.png)

- 数据库
```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
print(db.run("SELECT * FROM Artist LIMIT 10;"))
```
- Chain
```python
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
# 测试以确保有效
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there"})
print(response)
```
```python
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
# 使用 QuerySQLDatabaseTool 将查询执行添加到我们的链中
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
chain.invoke({"question": "How many employees are there"})
```
```python
# 将自动生成和执行查询的方法，将问题和结果传递给LLM
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

chain.invoke({"question": "How many employees are there"})
```
- Agent
```python
# 初始化代理
from langchain_community.agent_toolkits import create_sql_agent

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
# 执行测试
agent_executor.invoke(
    {
        "input": "List the total sales per country. Which country's customers spent the most?"
    }
)
agent_executor.invoke({"input": "Describe the playlisttrack table"})

```

[demo02.ipynb](LangChain%2Fdemo02.ipynb)

### Reference(参考文档)

* [LangChain-SQL](https://python.langchain.com/docs/use_cases/sql/quickstart)



