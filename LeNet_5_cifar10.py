# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:05:58 2024

@author: lich5
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载CIFAR - 10数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 对数据进行预处理
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = tf.reshape(x_train, [-1, 28, 28, 3])
x_train = tf.pad(x_train, [[0,0],[2,2],[2,2],[0,0]], 'CONSTANT')
x_test = tf.reshape(x_test, [-1, 28, 28, 3])
x_test = tf.pad(x_test, [[0,0],[2,2],[2,2],[0,0]], 'CONSTANT')

# 创建模型 
model = tf.keras.Sequential()
'''卷积层'''
model.add(tf.keras.layers.Input(shape=(32,32,3)))
model.add(tf.keras.layers.Conv2D(16, kernel_size=(5,5), activation='relu', 
                                 kernel_initializer='Orthogonal', padding='valid'))
# 创建模型
model = tf.keras.Sequential()
'''输入层'''
model.add(tf.keras.layers.Input(shape=(32,32,1)))
'''卷积层C1：核大小5x5，深度6，步长1（默认值），填充0（默认值）'''
model.add(tf.keras.layers.Conv2D(6, kernel_size=5, activation='sigmoid'))
'''池化层S2：窗口大小：2x2， 步长：2'''
model.add(tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2))
'''卷积层C3：核大小5x5，深度16，步长1（默认值），填充0（默认值）'''
model.add(tf.keras.layers.Conv2D(16, kernel_size=5, activation='sigmoid'))
'''池化层S4：窗口大小：2x2， 步长：2'''
model.add(tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2))
'''拉平数据'''
model.add(tf.keras.layers.Flatten())
'''全连接C5：120个神经元'''
model.add(tf.keras.layers.Dense(units=120,activation='sigmoid'))
'''全连接F6：84个神经元'''
model.add(tf.keras.layers.Dense(units=84,activation='sigmoid'))
'''全连接F6：10个神经元'''
model.add(tf.keras.layers.Dense(units=10,activation='softmax'))

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 评估模型
model.evaluate(x_test, y_test, verbose=2)

# 随机选择一些样本进行可视化
idx = np.random.randint(len(x_test), size=16)
images = x_test.numpy().squeeze()[idx,:]
y_ = y_test[idx]

# 测试模型
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis = 1)

# 绘图函数调整为显示CIFAR - 10彩色图像
def plot_cifar_44(images, y_, y=None):
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

plot_cifar_44(images, y_, y_pred[idx])

# [EOF]