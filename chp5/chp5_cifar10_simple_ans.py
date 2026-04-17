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

# 创建模型
# 这里使用简单的卷积神经网络示例，可根据需求调整
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', #选择算法 SGD
              loss='sparse_categorical_crossentropy',# 选择损失函数
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 评估模型
model.evaluate(x_test, y_test, verbose=2)

# 随机选择一些样本进行可视化
idx = np.random.randint(len(x_test), size=9)
images = x_test[idx]
y_ = y_train[idx]

# 测试模型
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis = 1)

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
