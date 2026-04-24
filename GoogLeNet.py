# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:05:29 2024

@author: lich5
"""

import numpy as np # linear algebra
import tensorflow as tf
# from tensorflow import keras
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, Model, Sequential, datasets
from tensorflow.keras.layers import MaxPool2D
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


class Inception(tf.keras.Model):
    # c1--c4是每条路径的输出通道数
    def __init__(self, ch1x1, ch3x3, ch5x5, pool_proj):
        super().__init__()
        # 线路1，单1x1卷积层
        self.p1_1 = layers.Conv2D(ch1x1, 1, activation='relu')
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = layers.Conv2D(ch3x3[0], 1, activation='relu')
        self.p2_2 = layers.Conv2D(ch3x3[1], 3, padding='same',
                                           activation='relu')
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = layers.Conv2D(ch5x5[0], 1, activation='relu')
        self.p3_2 = layers.Conv2D(ch5x5[1], 5, padding='same',
                                           activation='relu')
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = layers.Conv2D(pool_proj, 1, activation='relu')


    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维度上连结输出
        return layers.Concatenate()([p1, p2, p3, p4])
    

class InceptionAux(tf.keras.Model):
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
    

# class GoogLeNet(im_height=224, im_width=224, class_num=1000, aux_logits=False):
#     # tensorflow中的tensor通道排序是NHWC
#     input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    
#     # def b1:
#     # (None, 224, 224, 3)
#     x = layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", activation="relu")(input_image)
#     # (None, 112, 112, 64)
#     x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)
    
#     # def b2:
#     # (None, 56, 56, 64)
#     x = layers.Conv2D(64, kernel_size=1, activation="relu")(x)
#     # (None, 56, 56, 64)
#     x = layers.Conv2D(192, kernel_size=3, padding="SAME", activation="relu")(x)
#     # (None, 56, 56, 192)
#     x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    
#     # def b3:
#     # (None, 28, 28, 192)
#     x = Inception(64, (96, 128), (16, 32), 32)(x)
#     # (None, 28, 28, 256)
#     x = Inception(128, (128, 192), (32, 96), 64)(x)
#     # (None, 28, 28, 480)
#     x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)
#     # (None, 14, 14, 480)
    
#     # def b4:
#     x = Inception(192, (96, 208), (16, 48), 64)(x)
#     if aux_logits:
#         aux1 = InceptionAux(class_num)(x)

#     # (None, 14, 14, 512)
#     x = Inception(160, (112, 224), (24, 64), 64)(x)
#     # (None, 14, 14, 512)
#     x = Inception(128, (128, 256), (24, 64), 64)(x)
#     # (None, 14, 14, 512)
#     x = Inception(112, (144, 288), (32, 64), 64)(x)
#     if aux_logits:
#         aux2 = InceptionAux(class_num)(x)

# # # def b5:
#     # (None, 14, 14, 528)
#     x = Inception(256, (160, 320), (32, 128), 128)(x)
#     # (None, 14, 14, 532)
#     x = Inception(384, (192, 384), (48, 128), 128)(x)
#     # (None, 7, 7, 1024)
#     x = layers.GlobalAvgPool2D()(x)
#     # (None, 1, 1, 1024)
#     x = layers.Flatten()(x)
#     x = layers.Dense(class_num)(x)
#     # (None, class_num)
#     aux3 = layers.Softmax(x)

#     if aux_logits:
#         model = models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
#     else:
#         model = models.Model(inputs=input_image, outputs=aux3)
#     return model


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
    im_height = 96
    im_width = 96
    batch_size = 128
    epochs = 3
    
    # model = GoogLeNet(im_height=im_height, im_width=im_width, class_num=10, aux_logits=True)
    model = tf.keras.Sequential()
    
# def b1:
    model.add(layers.Conv2D(64, 7, strides=2, padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
    
# def b2:
    model.add(layers.Conv2D(64, 1, activation='relu'))
    model.add(layers.Conv2D(192, 3, padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
    
# def b3:
    model.add(Inception(64, (96, 128), (16, 32), 32))
    model.add(Inception(128, (128, 192), (32, 96), 64))
    model.add(layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
    
# def b4:
    model.add(Inception(192, (96, 208), (16, 48), 64))
    model.add(Inception(160, (112, 224), (24, 64), 64))
    model.add(Inception(128, (128, 256), (24, 64), 64))
    model.add(Inception(112, (144, 288), (32, 64), 64))
    model.add(Inception(256, (160, 320), (32, 128), 128))
    model.add(layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

# def b5:
    model.add(Inception(256, (160, 320), (32, 128), 128))
    model.add(Inception(384, (192, 384), (48, 128), 128))
    model.add(layers.GlobalAvgPool2D())
    model.add(layers.Flatten())

# def FC
    model.add(layers.Dense(10))
    
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