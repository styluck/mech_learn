# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:12:01 2024

@author: lich5
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

train_ds=tf.data.Dataset.from_tensor_slices((train_images,train_labels))
test_ds=tf.data.Dataset.from_tensor_slices((test_images,test_labels))

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plt.figure(figsize=(30,30))
# for i,(image,label) in enumerate(train_ds.shuffle(100000).take(20)):
#     #print(label)
#     ax=plt.subplot(5,5,i+1)
#     plt.imshow(image)
#     plt.title(CLASS_NAMES[label.numpy()[0]])
#     plt.axis('off')

def process_image(image,label):
    image=tf.image.per_image_standardization(image)
    image=tf.image.resize(image,
                          (224,224),
                          method=tf.image.ResizeMethod.BILINEAR)
    
    return image,label

train_ds_size=tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size=tf.data.experimental.cardinality(test_ds).numpy()

train_ds=(train_ds
          .map(process_image)
          .shuffle(buffer_size=train_ds_size)
          .batch(batch_size=128,drop_remainder=True)
         )
test_ds=(test_ds
          .map(process_image)
          .shuffle(buffer_size=test_ds_size)
          .batch(batch_size=128,drop_remainder=True)
         )

model = tf.keras.Sequential()
model.add(tf.keras.layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear")) #, input_shape=x_train.shape[1:]
model.add(tf.keras.layers.Conv2D(96, 11, strides=4, activation='relu', padding='same'))
model.add(tf.keras.layers.Lambda(tf.nn.local_response_normalization))
#Lambda层可以把任意的一个表达式作为一个“Layer”对象
#Lambda层之所以存在是因为它可以在构建Squential时使用任意的函数或者说tensorflow 函数
model.add(tf.keras.layers.MaxPooling2D(3, strides=2))
model.add(tf.keras.layers.Conv2D(256, 5, strides=1, activation='relu', padding='same'))
model.add(tf.keras.layers.Lambda(tf.nn.local_response_normalization))
model.add(tf.keras.layers.MaxPooling2D(3, strides=2))
model.add(tf.keras.layers.Conv2D (384, 3, strides=1, activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(384, 3, strides=1, activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(256, 3, strides=1, activation='relu', padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=1e-2, 
                                momentum=0.9,
                                weight_decay=5e-4),
    metrics=['accuracy']    
)
# model.summary()
# model.history.history.keys()

history=model.fit(
    train_ds,
    epochs=10, #50
    validation_data=test_ds
)


# 保存模型
model.save('cnn_model.h5')

# 加载模型
model = tf.keras.models.load_model('cnn_model.h5')

# model.evaluate(test_ds, verbose=2)

idx = np.random.randint(1e4,size=9)
images = test_images[idx,:]
y_ = test_labels[idx]

# 测试模型
def plot_cifar10_3_3(images, y_, y=None):
    assert images.shape[0] == len(y_)
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='binary')
        if y is None:
            xlabel = 'True: {}'.format(CLASS_NAMES[y_[i][0]])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(CLASS_NAMES[y_[i][0]], CLASS_NAMES[y[i]])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
'''利用predict命令，输入x_test生成测试样本的测试值'''
predictions = model.predict(images)
y_pred = np.argmax(predictions, axis = 1)

plot_cifar10_3_3(images, y_, y_pred)

f,ax=plt.subplots(2,1,figsize=(10,10)) 

#Assigning the first subplot to graph training loss and validation loss
ax[0].plot(history.history['loss'],color='b',label='Training Loss')
ax[0].plot(history.history['val_loss'],color='r',label='Validation Loss')
plt.legend()

#Plotting the training accuracy and validation accuracy
ax[1].plot(history.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(history.history['val_accuracy'],color='r',label='Validation Accuracy')

plt.legend()