### Chroma
开源的向量数据库，旨在为开发人员和各种规模的组织提供构建基于大型语言模型（LLM）的应用所需的资源。
```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 加载音频文件
y, sr = librosa.load('audio_file.mp3')

# 提取色度特征
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# 显示色度特征
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()
```
### Pinecone
Pinecone是闭源的
SaaS向量数据库，用于构建和部署大规模相似性搜索应用程序。它允许用户存储和查询向量数据
```python
import pinecone

# 配置你的 Pinecone API key
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# 创建一个新的索引
index_name = "example-index"
pinecone.create_index(index_name, dimension=128)

# 连接到你的索引
index = pinecone.Index(index_name)

# 向索引中添加一些向量
vectors = [("item1", [1, 2, 3, ...]), ("item2", [2, 3, 4, ...]), ...]  # 128-dimension vectors
index.upsert(vectors=vectors)

# 查询索引
query_results = index.query(queries=[[1, 2, 3, ...]], top_k=3)

# 打印查询结果
print(query_results)

# 删除索引
pinecone.delete_index(index_name)
```

### Faiss
由 Facebook AI Research (FAIR) 开发的一个库
用于高效地搜索大规模向量集合并提供快速的相似性搜索
```python
import faiss
import numpy as np

# 假设我们有一个向量集合，我们想要在其中进行搜索
d = 64                           # 向量维度
nb = 100000                      # 数据库大小
np.random.seed(1234)             
db_vectors = np.random.random((nb, d)).astype('float32')
db_vectors[:, 0] += np.arange(nb) / 1000.

# 构建索引
index = faiss.IndexFlatL2(d)    # 使用L2距离建立索引
index.add(db_vectors)            # 向索引中添加向量

# 现在我们可以进行搜索
nq = 10                          # 查询向量的数量
np.random.seed(1234)             
query_vectors = np.random.random((nq, d)).astype('float32')
query_vectors[:, 0] += np.arange(nq) / 1000.

k = 4                            # 我们想要检索的最近邻的数量
distances, indices = index.search(query_vectors, k)  # 实际的搜索

# distances 和 indices 分别保存了查询的距离和索引
print(indices)
print(distances)
```




### Reference(参考文档)

* [2023年排名前五的向量数据库](https://juejin.cn/post/7251223932232171575)




