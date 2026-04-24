# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:00:13 2024

@author: lich5
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:23:08 2024

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


class Residual(Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = layers.Conv2D(
            num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = layers.Conv2D(
            num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = layers.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
    
    
class ResnetBlock(layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X


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
    
    model = Sequential()

    '''完成ResNet'''
    #################################
    # def b1:
    
    # def b2~b5:
    
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