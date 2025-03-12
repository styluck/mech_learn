# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:43:06 2024

@author: lich5
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

# 加载数据
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 数据预处理函数
def preprocess(x, y):
    x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1  # 归一化到[-1, 1]
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    return x, y

# 处理训练集和测试集
x_train, y_train = preprocess(x_train, y_train)
x_test, y_test = preprocess(x_test, y_test)

# 构建模型
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 打平输入
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])

# 定义优化器
optimizer = optimizers.Adam(learning_rate=0.001)

# 构建数据集
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(60000).batch(512)

# 训练参数
epochs = 5
train_loss_history = []

# 创建可视化画布

# 训练循环
for epoch in range(epochs):
    epoch_loss = []
    
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # 前向传播
            out = model(x_batch)
            # 计算损失
            y_onehot = tf.one_hot(y_batch, depth=10)
            loss = tf.reduce_mean(tf.square(out - y_onehot))
        
        # 反向传播
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        epoch_loss.append(loss.numpy())
        
        # 每100个batch打印进度
        if step % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.numpy():.4f}")
    
    # 记录每个epoch的平均损失
    train_loss_history.append(tf.reduce_mean(epoch_loss).numpy())

# 绘制训练损失曲线
plt.figure(figsize=(7, 5))
plt.plot(range(1, epochs+1), train_loss_history, 'o-')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 测试数据可视化
sample_images = x_test[:16]
sample_labels = y_test[:16]

# 模型预测
predictions = model(sample_images)
pred_labels = tf.argmax(predictions, axis=1).numpy()
plt.figure(figsize=(12, 8))

# 可视化展示
# for i in range(5):
#     plt.subplot(1, 5, i+1)
#     img = (sample_images[i].numpy() + 1) / 2  # 反归一化到[0,1]
#     plt.imshow(img, cmap='gray')
#     plt.title(f"Pred: {pred_labels[i]}\nTrue: {sample_labels[i].numpy()}")
#     plt.axis('off')

# plt.tight_layout()
# plt.show()
for i in range(16):
    plt.subplot(4, 4, i+1)  # 4行4列
    img = (sample_images[i].numpy() + 1) / 2  # 反归一化到[0,1]
    
    plt.imshow(img, cmap='gray')  # 显示图像
    
    # 标注预测结果（绿色正确，红色错误）
    true_label = sample_labels[i].numpy()
    pred_label = pred_labels[i]
    color = 'green' if pred_label == true_label else 'red'
    
    plt.title(f"P:{pred_label}\nT:{true_label}", 
              color=color, 
              fontsize=9)  
    plt.axis('off')

plt.suptitle("Model Predictions (P=Prediction, T=True Label)", 
             y=0.92, 
             fontsize=12)
plt.tight_layout()  
plt.show()

# 计算测试准确率
test_pred = model(x_test)
y_test = tf.cast(y_test, tf.int64)
test_acc = tf.reduce_mean(
    tf.cast(tf.argmax(test_pred, axis=1) == y_test, tf.float32)
)
print(f"\nTest Accuracy: {test_acc.numpy()*100:.2f}%")

# [EOF]