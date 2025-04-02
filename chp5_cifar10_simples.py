# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:05:58 2024

@author: lich5
练习：完成代码，并记录各项测试的结果。
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载CIFAR - 10数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 数据预处理
'''
对数据进行归一化预处理
'''

# 创建模型
'''
利用keras.Sequential命令创建神经网络
1. 在模型中添加一个输出为128的全连接层，测试使用ReLU、GeLU、ELU、LeakyReLU的效果
2. 在模型中添加一个输出为10的全连接层，测试使用softmax、tanh的效果
''' 

# 编译模型
'''
利用compile命令对模型进行编译
损失函数使用sparse_categorical_crossentropy
测试使用SGD、dadgrad、RMSProp、adam算法作为优化器的效果
''' 

# 训练模型
'''
利用fit命令训练模型，添加(x_test, y_test)作为validation_data
训练10个epoch
'''

# 评估模型
'''
使用evaluate命令评估模型
'''


#%% 随机选择一些样本进行可视化
idx = np.random.randint(len(x_test), size=9)
images = x_test[idx]
y_ = y_train[idx]

# 测试模型
'''
用predict命令，添加x_test数据测试模型，生成y_pred数据
'''
y_pred = 0

# 绘图函数调整为显示CIFAR - 10彩色图像
def plot_cifar_3_3(images, y_, y=None):
    assert images.shape[0] == len(y_)
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        if y is None:
            xlabel = 'True: {}'.format(y_[i][0])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(y_[i][0], y[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

plot_cifar_3_3(images, y_, y_pred[idx])

# [EOF]