# 实例 pip install scikit-learn
from sklearn.cluster import KMeans
import numpy as np

# 创建示例数据
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# 定义K值
k = 2
# 初始化K均值聚类模型
kmeans = KMeans(n_clusters=k)
# 使用模型拟合数据
kmeans.fit(X)
# 输出簇中心
print("Cluster centers:\n", kmeans.cluster_centers_)
# 预测样本所属的簇
print("Cluster predictions:", kmeans.labels_)
