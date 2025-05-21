# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:44:38 2024

@author: lich5
Replicate: LeNet-5
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

'''加载 MNIST或cifar10 数据集''' 
x_train = 0
y_train = 0
x_test = 0
y_test = 0

'''对数据进行归一化预处理'''


'''利用pad操作将28*28*的图像拉成32*32的图像'''


'''创建模型'''
model = tf.keras.Sequential()

model.build()
model.summary()

'''编译模型，采用adam算法以及sparse_categorical_crossentropy损失'''


start = time.perf_counter()
'''用fit训练模型'''

end = time.perf_counter() # time.process_time()
c=end-start 
print("程序运行总耗时:%0.4f"%c, 's') 

'''用evaluate评估模型'''
y_=0

# 随机选择一些样本进行可视化
idx = np.random.randint(len(x_test), size=16)
images = x_test.numpy().squeeze()[idx,:]
y_ = y_test[idx]
 
# 测试模型
def plot_44(images, y_, y=None):
    assert images.shape[0] == len(y_)
    fig, axes = plt.subplots(4, 4)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape([32,32]), cmap='binary')
        if y is None:
            xlabel = 'True: {}'.format(y_[i])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(y_[i], y[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

'''利用predict命令，输入x_test生成测试样本的测试值''' 
y_pred = 0

plot_44(images, y_, y_pred[idx])

 # [EOF]


