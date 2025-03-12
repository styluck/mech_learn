# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:43:06 2024

@author: lich5
"""
import os
import tensorflow as tf # 导入 TF 库
# from tensorflow import keras # 导入 TF 子库 keras
from tensorflow.keras import layers, optimizers, datasets # 导入 TF 子库等

# 加载数据
(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1  # 归一化到[-1, 1]
y = tf.convert_to_tensor(y, dtype=tf.int32)

# 构建模型
model = tf.keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

# 定义优化器 
optimizer = optimizers.Adam(learning_rate=0.001)  # 使用Adam优化器

# 构建数据集
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(512)

# 训练循环
for epoch in range(5):  # 训练5个epoch
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        # 前向计算与梯度记录
        with tf.GradientTape() as tape:
            x_batch = tf.reshape(x_batch, (-1, 28 * 28))  # 打平输入
            out = model(x_batch)                        # 前向传播
            y_onehot = tf.one_hot(y_batch, depth=10)    # one-hot编码
            loss = tf.reduce_mean(tf.square(out - y_onehot))  # 均方误差

        # 计算梯度
        grads = tape.gradient(loss, model.trainable_variables)
        
        # 更新参数（关键修正：使用优化器实例）
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 打印训练信息
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.numpy()}")


