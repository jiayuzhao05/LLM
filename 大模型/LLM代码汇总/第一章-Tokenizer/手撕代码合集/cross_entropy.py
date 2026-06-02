# BCE loss实现（二分类交叉熵）
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(x))
def cross_entropy_error(p, y):
    delta = 1e-7  # 添加一个微小值，防止负无穷(np.log(0))的情况出现
    p = sigmoid(p)
    return -np.sum(y * np.log(p + delta) + (1 - y) * np.log(1 - p + delta))
def backward(p, y):
    return np.sum(sigmoid(p) - y)

# -----------------------------------------------------------------

# 多分类交叉熵实现
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)
def cross_entropy_error(p, y):
    delta = 1e-7  # 添加一个微小值，防止负无穷(np.log(0))的情况出现
    p = softmax(p)
    return -np.sum(y * np.log(p + delta))
def backward(p, y):
    return np.sum(y*(softmax(p) - 1))