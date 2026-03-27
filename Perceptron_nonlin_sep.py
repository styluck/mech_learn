# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:06:42 2025

@author: lich5
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 生成异或问题的数据
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
y = np.where(y == 0, -1, 1)  # 将标签转换为 -1 和 1

# 初始化感知器的权重和偏置
w = np.zeros(2)
b = 0

# 学习率
learning_rate = 0.1
# 最大迭代次数
max_iter = 100

# 记录每次迭代的权重和偏置
weights_history = []
biases_history = []

# 感知器学习过程
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
    weights_history.append(w.copy())
    biases_history.append(b)

# 设置图形
fig, ax = plt.subplots()
ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='b', label='Class -1')
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='r', label='Class 1')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
line, = ax.plot([], [], 'g-', label='Decision Boundary')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Perceptron Training on Non - linearly Separable Data')
ax.legend()

# 初始化函数
def init():
    line.set_data([], [])
    return line,

# 更新函数
def update(frame):
    w = weights_history[frame]
    b = biases_history[frame]
    x_range = np.linspace(xlim[0], xlim[1], 100)
    if w[1] != 0:
        y_range = -(w[0] * x_range + b) / w[1]
        line.set_data(x_range, y_range)
    return line,

# 创建动画
ani = FuncAnimation(fig, update, frames=max_iter, init_func=init, blit=True, interval=100)

plt.show()

# [EOF]