# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:44:08 2025

@author: lich5
"""

import numpy as np
import matplotlib.pyplot as plt

# 生成一些简单的二分类数据
np.random.seed(42)
# 类别1的数据
X1 = np.random.randn(10, 2) + [2, 2]
y1 = np.ones(X1.shape[0])
# 类别 -1的数据
X2 = np.random.randn(10, 2) + [-2, -2]
y2 = -np.ones(X2.shape[0])

# 合并数据
X = np.vstack((X1, X2))
Y = np.hstack((y1, y2))

# 打乱数据顺序
shuffle_idx = np.random.permutation(len(Y))
X = X[shuffle_idx]
Y = Y[shuffle_idx]

# 设置一个比较差的初始点
# 这里将权重初始化为一个较大的负数，偏置初始化为一个较大的正数
w = np.array([-30,30])
b = 0

'''调整学习率和最大迭代次数'''
learn_rate = 1.5# 0.1
maxiter = 40

# 记录每次迭代的权重和偏置
weights_history = []
biases_history = []
current_sample_history = []

weights_history.append(w.copy())
biases_history.append(b)
# current_sample_history.append(X[0])

# 感知器学习算法
for iter in range(maxiter):
    misclassified = 0
    for i in range(len(Y)):
        # 计算预测值
        prediction = np.dot(w, X[i]) + b
        if Y[i] * prediction <= 0:
            # 如果预测错误，更新权重和偏置
            w = w + learn_rate * Y[i] * X[i]
            b = b + learn_rate * Y[i]
            misclassified += 1
            
            current_sample_history.append(X[i])
            weights_history.append(w.copy())
            biases_history.append(b)
    if misclassified == 0 and iter > 9:
        print(f"Converged after {iter + 1} iterations.")
        break

# 计算子图的行数和列数
nrows = int(np.ceil(np.sqrt(len(current_sample_history))))
ncols = int(np.ceil(len(current_sample_history) / nrows))

fig, axes = plt.subplots(ncols, nrows, figsize=(8, 8))
axes = axes.flatten()

x_min = -6
x_max = 6
y_min = -6
y_max = 6

x_range = np.linspace(x_min, x_max, 100)
for i in range(len(current_sample_history)-1):
    w = weights_history[i]
    b = biases_history[i]
    ax = axes[i]
    ax.scatter(X1[:, 0], X1[:, 1], c='b', label='Class 1')
    ax.scatter(X2[:, 0], X2[:, 1], c='r', label='Class -1')
    if w[1] != 0:
        y_range = -(w[0] * x_range + b) / w[1]
        ax.plot(x_range, y_range, 'g', label=f'Iteration {i + 1} Decision Boundary')
        
    current_sample = current_sample_history[i]
    ax.scatter(current_sample[0], current_sample[1], c='yellow', s=20,  label='Current Sample')
    # 绘制权重向量 w
    arrow_length = .05  # 向量长度
    ax.arrow(0, 0, w[0] * arrow_length, w[1] * arrow_length, head_width=1, head_length=1, fc='orange', ec='orange', label='Weight Vector w')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Iteration {i + 1}')
    # ax.legend()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

for j in range(len(weights_history), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# [EOF]