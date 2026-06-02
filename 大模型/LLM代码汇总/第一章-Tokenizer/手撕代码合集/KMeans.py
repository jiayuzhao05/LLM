import numpy as np
 
class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
    def fit(self, X):
        # 初始化质心
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        centroids = X[indices]
        for _ in range(self.max_iters):
            # 将每个点分配给最近的质心
            clusters = [[] for _ in range(self.k)]
            for features in X:
                distances = [np.linalg.norm(features - centroid) for centroid in centroids]
                closest_cluster = np.argmin(distances)
                clusters[closest_cluster].append(features)
            # 计算新的质心
            new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters if cluster])
            # 检查质心是否变化
            if np.allclose(centroids, new_centroids, atol=self.tol):
                break
            centroids = new_centroids
        self.centroids = centroids
        self.clusters = clusters
    def predict(self, X):
        y_pred = [np.argmin([np.linalg.norm(x - centroid) for centroid in self.centroids]) for x in X]
        return np.array(y_pred)
# 示例使用
if __name__ == "__main__":
    # 生成一些随机数据
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
 
    # 创建KMeans实例并拟合数据
    kmeans = KMeans(k=4)
    kmeans.fit(X)
 
    # 预测每个点的簇标签
    y_pred = kmeans.predict(X)
 
    # 打印质心
    print("Centroids:")
    print(kmeans.centroids)
 
    # 你可以使用matplotlib来可视化结果
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, alpha=0.75)
    plt.show()