# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:00:57 2024

@author: lich5
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras import layers, models, Model, Sequential, datasets
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


class Inception(Model):
    # c1--c4是每条路径的输出通道数
    def __init__(self, ch1x1, ch3x3, ch5x5, Pool):
        super().__init__()
        # 线路1，单1x1卷积层
        self.p1_1 = layers.Conv2D(ch1x1, 1, activation='relu')
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = layers.Conv2D(ch3x3[0], 1, activation='relu')   
        self.p2_2 = layers.Conv2D(ch3x3[1], 3, padding='same', activation='relu')        
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = layers.Conv2D(ch5x5[0], 1, activation='relu')   
        self.p3_2 = layers.Conv2D(ch5x5[1], 5, padding='same', activation='relu')   
        
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = layers.MaxPool2D(3,1, padding='same')
        self.p4_2 = layers.Conv2D(Pool, 1, activation='relu')
        
    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维度上连结输出
        return layers.Concatenate()([p1, p2, p3, p4])
    

class InceptionAux(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.averagePool = layers.AvgPool2D(pool_size=5, strides=3)
        self.conv = layers.Conv2D(128, kernel_size=1, activation="relu")

        self.fc1 = layers.Dense(1024, activation="relu")
        self.fc2 = layers.Dense(num_classes)
        self.softmax = layers.Softmax()

    def call(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 2048
        x = self.fc1(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        x = self.softmax(x)

        return x
    

if __name__ == '__main__':

    
#%% load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    train_ds=tf.data.Dataset.from_tensor_slices((train_images,train_labels))
    test_ds=tf.data.Dataset.from_tensor_slices((test_images,test_labels))
    
    CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 
                  'frog', 'horse', 'ship', 'truck']
    
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

#%% define the model
    im_height = 224
    im_width = 224
    batch_size = 128
    epochs = 3
    
    model = tf.keras.Sequential()
    
    '''完成单通道输出的GoogLeNet'''
    #################################
    # def b1:
    model.add(Inception(64, (96,128),(16,32),32))
    # def b2:
        
    # def b3:
        
    # def b4:
    
    # def b5:
    
    # def FC:
    
    #################################

    # compile
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.optimizers.Adam(learning_rate=0.0005),
        metrics=['accuracy']    
    )
    # model.build((batch_size, 224, 224, 3))  # when using subclass model
    # model.summary()
    
    history=model.fit(
        train_ds,
        epochs=epochs, #50
        validation_data=test_ds
    )
    
    
    # # 保存模型
    # model.save('cnn_model.h5')
    
    # # 加载模型
    # model = tf.keras.models.load_model('cnn_model.h5')

    model.evaluate(test_ds, verbose=2)
    
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
    
    #Plotting the training accuracy and validation accuracy
    ax[1].plot(history.history['accuracy'],color='b',label='Training  Accuracy')
    ax[1].plot(history.history['val_accuracy'],color='r',label='Validation Accuracy')
    
    plt.legend()





 # [EOF]
