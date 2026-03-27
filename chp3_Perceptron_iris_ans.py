# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:48:11 2025

@author: lich5
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D

'''# 加载鸢尾花数据集'''
iris = load_iris()
# 选取前两个类别构建二分类问题
X = iris.data[:100, :2]
y = iris.target[:100]
# 将标签转换为 -1 和 1
y = np.where(y == 0, -1, 1)

# 打乱数据顺序
shuffle_idx = np.random.permutation(len(y))
X = X[shuffle_idx]
y = y[shuffle_idx]

'''# 设置初始权重和偏置'''
w = np.random.randn(X.shape[1]) * -30
b = 0

'''调整学习率和最大迭代次数'''
learning_rate = 0.5
max_iter = 2000

# 记录每次迭代的权重、偏置和误分类样本数量
weights_history = []
biases_history = []
misclassified_history = []
current_sample_history = []

weights_history.append(w.copy())
biases_history.append(b)

'''# 载入感知器学习算法，参考Perceptron.py写一个'''
for iter in range(max_iter):
    misclassified = 0
    for i in range(len(y)):
        # 计算预测值
        prediction = np.dot(w, X[i]) + b
        if y[i] * prediction <= 0:
            # 如果预测错误，更新权重和偏置
            w = w + learning_rate * y[i] * X[i]
            b = b + learning_rate * y[i]
            misclassified += 1
    misclassified_history.append(misclassified)
    weights_history.append(w.copy())
    biases_history.append(b)
    if misclassified == 0 and iter >20 :
        print(f"Converged after {iter + 1} iterations.")
        break


# 测试模型（这里使用全部数据作为测试集）
correct_predictions = 0
predictions = []
for i in range(len(y)):
    prediction = np.dot(w, X[i]) + b
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

# 修正截距
# 计算其他特征的均值
x3_mean = 0#np.mean(X[:, 2])
x4_mean = 0#np.mean(X[:, 3])
# b_prime = (b + w[2] * x3_mean + w[3] * x4_mean) / w[eig2]

b_prime = (b) / w[eig2]

x_range = np.linspace(np.min(X[:, eig1]), np.max(X[:, eig1]), 100)
if w[eig2] != 0:
    y_range = -(w[eig1] / w[eig2]) * x_range - b_prime
    plt.plot(x_range, y_range, 'g', label='Decision Boundary')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Decision Boundary in Feature Space')
plt.legend()

plt.tight_layout()
# plt.show()
# # 可视化训练过程中的误分类样本数量
# plt.figure(figsize=(6, 5))
# plt.plot(misclassified_history)
# plt.title('Number of Misclassified Samples per Iteration')
# plt.xlabel('Iteration')
# plt.ylabel('Number of Misclassified Samples')

# # 可视化特征空间中的决策边界（仅选取前两个特征进行可视化）
# # 选取前三个特征用于三维可视化
# eig1 = 0
# eig2 = 1
# eig3 = 2

# # 创建 3D 图形
# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(111, projection='3d')

# # 绘制数据点
# ax.scatter(X[y == -1][:, eig1], X[y == -1][:, eig2], X[y == -1][:, eig3], c='b', label='Class -1')
# ax.scatter(X[y == 1][:, eig1], X[y == 1][:, eig2], X[y == 1][:, eig3], c='r', label='Class 1')

# # 计算其他特征的均值（这里假设使用第四个特征）
# x4_mean = np.mean(X[:, 3])

# # 决策平面方程：w[0]*x + w[1]*y + w[2]*z + w[3]*x4_mean + b = 0
# # 解出 z：z = -(w[0]*x + w[1]*y + w[3]*x4_mean + b) / w[2]

# x_range = np.linspace(np.min(X[:, eig1]), np.max(X[:, eig1]), 100)
# y_range = np.linspace(np.min(X[:, eig2]), np.max(X[:, eig2]), 100)
# X_grid, Y_grid = np.meshgrid(x_range, y_range)

# if w[eig3] != 0:
#     Z_grid = -(w[eig1] * X_grid + w[eig2] * Y_grid + w[3] * x4_mean + b) / w[eig3]
#     ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.5, color='g', label='Decision Boundary')

# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Feature 3')
# ax.set_title('3D Decision Boundary in Feature Space')
# # ax.legend()

# plt.show()

# [EOF]