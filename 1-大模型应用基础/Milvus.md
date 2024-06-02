### Milvus

```python
client = MilvusClient(
    uri=config.MilvusClientUrl,
    token=config.MILVUS_API_TOKEN
)
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
schema.add_field(field_name="my_text", datatype=DataType.VARCHAR, max_length=max_length)
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="my_vector",
    index_type="AUTOINDEX",
    metric_type=metric_type
)
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)
client.insert(
    collection_name=collection_name,
    data=chunks[i:i + 99]
)
```

### Chroma

开源的向量数据库，旨在为开发人员和各种规模的组织提供构建基于大型语言模型（LLM）的应用所需的资源。

### Pinecone

Pinecone是闭源的
SaaS向量数据库，用于构建和部署大规模相似性搜索应用程序。它允许用户存储和查询向量数据

### Faiss

由 Facebook AI Research (FAIR) 开发的一个库
用于高效地搜索大规模向量集合并提供快速的相似性搜索

### Reference(参考文档)

* [2023年排名前五的向量数据库](https://juejin.cn/post/7251223932232171575)
