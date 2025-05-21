# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:11:52 2024

@author: lich5
可尝试STL-10、Caltech-101数据集
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar100.load_data(label_mode='fine')

train_ds=tf.data.Dataset.from_tensor_slices((train_images,train_labels))
test_ds=tf.data.Dataset.from_tensor_slices((test_images,test_labels))
class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

def process_image(image,label):
    image=tf.image.per_image_standardization(image)
    image=tf.image.resize(image,(64,64))
    
    return image,label



'''
请补充模型
'''


'''
请训练模型
'''
start = time.perf_counter()

history=model.fit(
    ?,
    epochs=1, #50
    validation_data=?
)


end = time.perf_counter() # time.process_time()
c=end-start 
print("程序运行总耗时:%0.4f"%c, 's') 


# 保存模型
model.save('my_alexnet_model.h5')

# 加载模型
model = tf.keras.models.load_model('my_alexnet_model.h5')

model.evaluate(?, verbose=2)

idx = np.random.randint(1e4,size=9)
images = test_images[idx,:]
y_ = test_labels[idx]

# 测试模型
def plot_cifar100_44(images, y_, y=None):
    assert images.shape[0] == len(y_)
    fig, axes = plt.subplots(4, 4)
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
predictions = model.predict(?)
y_pred = np.argmax(predictions, axis = 1)

plot_cifar100_44(images, y_, y_pred)

f,ax=plt.subplots(2,1,figsize=(10,10)) 

#Assigning the first subplot to graph training loss and validation loss
ax[0].plot(model.history.history['loss'],color='b',label='Training Loss')
ax[0].plot(model.history.history['val_loss'],color='r',label='Validation Loss')

#Plotting the training accuracy and validation accuracy
ax[1].plot(model.history.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(model.history.history['val_accuracy'],color='r',label='Validation Accuracy')

plt.legend()

#[EOF]
