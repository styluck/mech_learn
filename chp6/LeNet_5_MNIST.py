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

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 对数据进行预处理
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.reshape(x_train, [-1, 28, 28, 1])
x_train = tf.pad(x_train, [[0,0],[2,2],[2,2],[0,0]], 'CONSTANT')
x_test = tf.reshape(x_test, [-1, 28, 28, 1])
x_test = tf.pad(x_test, [[0,0],[2,2],[2,2],[0,0]], 'CONSTANT')

# train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
# test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test))

# # train_ds = train_ds.take(10000).shuffle(10000).batch(30)
# train_ds = train_ds.shuffle(10000).batch(30)
# test_ds = test_ds.shuffle(10000).batch(30)

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

model.build()
model.summary()

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
start = time.perf_counter()
model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

end = time.perf_counter() # time.process_time()
c=end-start 
print("程序运行总耗时:%0.4f"%c, 's') 

# 评估模型
model.evaluate(x_test, y_test, verbose=2)

# 随机选择一些样本进行可视化
idx = np.random.randint(len(x_test), size=16)
images = x_test.numpy().squeeze()[idx,:]
y_ = y_test[idx]
 
# 测试模型
def plot_mnist_44(images, y_, y=None):
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
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis = 1)

plot_mnist_44(images, y_, y_pred[idx])

 # [EOF]


