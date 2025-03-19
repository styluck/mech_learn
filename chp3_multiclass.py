# -*- coding: utf-8 -*-
"""
逻辑回归多分类可视化代码
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

# 设置全局字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_decision_boundary(model, X, y, title):
    """
    绘制决策边界和数据点的函数
    
    参数:
        model: 训练好的逻辑回归模型
        X: 特征矩阵
        y: 目标向量
        title: 图表标题
    """
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测概率并转换为类别标签
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.title(title)
    plt.axis('tight')
    
    # 绘制数据点
    colors = ['blue', 'red', 'yellow']
    for i, color in zip(range(3), colors):
        idx = np.where(y == i)[0]
        plt.scatter(X[idx, 0], X[idx, 1], c=color, edgecolor='k', s=20, label=f'Class {i+1}')
    
    # 绘制超平面
    def plot_hyperplane(class_idx, color):
        coef = model.coef_[class_idx]
        intercept = model.intercept_[class_idx]
        
        if abs(coef[1]) < 1e-9:
            # 垂直线
            x_val = -intercept / coef[0]
            plt.axvline(x_val, color=color, linestyle='--', label=f'Hyperplane {class_idx+1}')
        else:
            # 斜线
            x_vals = [x_min, x_max]
            y_vals = [(-coef[0]*x - intercept)/coef[1] for x in x_vals]
            plt.plot(x_vals, y_vals, color=color, linestyle='--', label=f'Hyperplane {class_idx+1}')
    
    for i in range(3):
        plot_hyperplane(i, colors[i])
    
    # 显示图例
    plt.legend()
    plt.show()

# 生成数据集
centers = [[-5, 0], [0, 1.5], [5, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)
transformation = [[0.4, 0.2], [-0.4, 1.2]]
X = np.dot(X, transformation)

# 比较两种多分类策略
for multi_class in ['multinomial', 'ovr']:
    # 训练模型
    clf = LogisticRegression(
        solver='sag', max_iter=100, 
        random_state=42, multi_class=multi_class
    ).fit(X, y)
    
    # 输出训练得分
    print(f"Training score ({multi_class}): {clf.score(X, y):.3f}")
    
    # 绘制结果
    plot_decision_boundary(clf, X, y, f"逻辑回归决策曲面 ({multi_class})")
    
# [EOF]