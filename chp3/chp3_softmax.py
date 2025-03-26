# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:43:06 2024

@author: lich5
"""
import os
import matplotlib.pyplot as plt

# 数据预处理函数
def preprocess(y,x):
    x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1  # 归一化到[-1, 1]
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    return x, y

'''# 加载mnist数据'''



'''# 处理训练集和测试集'''


'''# 构建模型'''


'''# 定义优化器'''


'''# 构建数据集'''


'''# 训练参数'''


'''# 训练循环'''


# 绘制训练损失曲线
plt.figure(figsize=(7, 5))
plt.plot(range(1, epos+1), train_his, 'o-')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 测试数据可视化
sample_images = X_test[:16]
sample_labels = Y_test[:16]

# 模型预测
predicts = my_model(sample_images)
pred_labels = tf.argmax(predicts, axis=1).numpy()
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
test_pred = model(X_test)
y_test = tf.cast(Y_test, tf.int64)
test_acc = tf.reduce_mean(
    tf.cast(tf.argmax(test_pred, axis=1) == y_test, tf.float32)
)
print(f"\nTest Accuracy: {test_acc.numpy()*100:.2f}%")

# [EOF]
