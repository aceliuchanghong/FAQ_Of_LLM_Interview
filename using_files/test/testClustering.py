import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(0)
X = np.random.rand(10, 2)

# 计算距离矩阵
Z = linkage(X, 'ward')

# 绘制聚类树
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.show()
