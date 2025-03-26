# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:54:41 2025

@author: lich5
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_iris

'''# 加载鸢尾花数据集'''


'''# 选取前两个类别构建二分类问题'''


'''# 将标签转换为 -1 和 1'''


'''# 创建 SVM 分类器实例，并训练模型'''


'''# 绘制数据点'''


# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

'''# 创建网格点'''


'''# 绘制决策边界和间隔边界'''


'''# 绘制支持向量'''


plt.title('SVM Binary Classification on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# [EOF]
