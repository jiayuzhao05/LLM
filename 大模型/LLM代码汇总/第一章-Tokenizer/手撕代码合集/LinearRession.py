#minimax一面code
import numpy as np

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(100, 2)  # 特征数据
y = 4 + 3 * X + np.random.randn(100, 1)  # 输出数据，添加了噪声

# 第一种，直接求出w
def linear_regression(X, y):
    ones = np.ones((X.shape[0], 1))
    X_b = np.concatenate([ones, X], axis=1)  # 添加x0=1的列
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best

theta_best = linear_regression(X, y)
print("最佳拟合参数:", theta_best)

# 第二种，使用SGD
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(), (predictions - y))
        descent = alpha * (1 / m) * error
        theta -= descent
        J_history[i] = compute_cost(X, y, theta)
        
    return theta, J_history

# 初始化参数
theta = np.random.randn(3, 2)

# 设置超参数
alpha = 0.01
num_iters = 1000

# 添加x0=1的列
ones = np.ones((X.shape[0], 1))
X_b = np.concatenate([ones, X], axis=1)

# 梯度下降
theta, _ = gradient_descent(X_b, y, theta, alpha, num_iters)
print("梯度下降后的参数:", theta)
# print("预测值:", X_b.dot(theta))    