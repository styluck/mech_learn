# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25  09:42:51 2025

@author: lich5
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


'''# 加载鸢尾花数据集'''
iris = load_iris()


# 选取前两个类别构建二分类问题
X = iris.data[:100]
y = iris.target[:100]
# 将标签转换为 -1 和 1
y = np.where(y == 0, -1, 1)

# 打乱数据顺序
shuffle_idx = np.random.permutation(len(y))
X = X[shuffle_idx]
y = y[shuffle_idx]

'''# 设置初始权重和偏置'''
W = np.random.randn(X.shape[1]) * -30
B = 0

'''调整学习率和最大迭代次数'''
learning_rate = 0.5
max_iter = 2000

# 记录每次迭代的权重、偏置和误分类样本数量
weights_history = []
biases_history = []
misclassified_history = []

weights_history.append(W.copy())
biases_history.append(B)

'''# 载入感知器学习算法，参考Perceptron.py写一个'''


# 测试模型（这里使用全部数据作为测试集）
correct_predictions = 0
predictions = []
for i in range(len(y)):
    prediction = np.dot(W, X[i]) + B
    if (prediction > 0 and y[i] == 1) or (prediction <= 0 and y[i] == -1):
        correct_predictions += 1
    predictions.append(1 if prediction > 0 else -1)

accuracy = correct_predictions / len(y)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# 可视化训练过程中的误分类样本数量
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(misclassified_history)
plt.title('Number of Misclassified Samples per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Number of Misclassified Samples')

# 可视化特征空间中的决策边界（仅选取前两个特征进行可视化）

# 可视化特征空间中的决策边界（仅选取前两个特征进行可视化）
eig1 = 0
eig2 = 1
plt.subplot(1, 2, 2)
plt.scatter(X[y == -1][:, eig1], X[y == -1][:, eig2], c='b', label='Class -1')
plt.scatter(X[y == 1][:, eig1], X[y == 1][:, eig2], c='r', label='Class 1')

# 计算其他特征的均值
x3_mean = np.mean(X[:, 2])
x4_mean = np.mean(X[:, 3])
# 修正截距
b_prime = (B + W[2] * x3_mean + W[3] * x4_mean) / W[eig2]

x_range = np.linspace(np.min(X[:, eig1]), np.max(X[:, eig1]), 100)
if W[eig2] != 0:
    y_range = -(W[eig1] / W[eig2]) * x_range - b_prime
    plt.plot(x_range, y_range, 'g', label='Decision Boundary')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Decision Boundary in Feature Space')
plt.legend()

plt.tight_layout()
plt.show()

# [EOF]