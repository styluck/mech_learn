# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:06:42 2025

@author: 6
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm 

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


# 创建SVM分类器实例
clf = svm.SVC(kernel='linear')
# 训练模型
clf.fit(X, Y)

# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired)

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

plt.title('SVM Binary Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
# [EOF]
