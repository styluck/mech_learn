# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:55:09 2025

@author: lich5
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()
# 选取前两个类别构建二分类问题
X = iris.data[:100, :2]  # 只选取前两个特征（萼片长度和萼片宽度）用于可视化
y = iris.target[:100]
# 将标签转换为 -1 和 1
y = np.where(y == 0, -1, 1)

# 创建 SVM 分类器实例
clf = svm.SVC(kernel='linear')
# 训练模型
clf.fit(X, y)

# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格点
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# 绘制决策边界和间隔边界
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# 绘制支持向量
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.title('SVM Binary Classification on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# [EOF]