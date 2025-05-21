# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:03:50 2024

@author: 6
"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        
        ############################################
        '''
        定义 encoder 和 decoder
        encoder网络：
        1. 输入的扁平化层
        2. 神经网络：可以使用全连接层、卷积层、循环层，神经元个数通过latent_dim变量定义
        '''         
        
        
        '''
        decoder网络：
        1. 神经网络：可以使用全连接层、卷积层、循环层，神经元个数通过shape变量定义
        2. 输出的Reshape层
        ''' 
        ############################################
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def psnr(X, Y):
    # Get the dimensions of the input image
    m, n = X.shape
    N = Y - X
        
    # Calculate the mean squared error (MSE)
    Pn1 = np.sum(N * N)
    Pn = Pn1 / (m * n)
    
    # Calculate PSNR
    PSNR = 10 * np.log10(1 / Pn)
    
    return PSNR


if __name__ == '__main__':
    
    # train the basic autoencoder using the Fashion MNIST dataset
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    
    shape = x_test.shape[1:]
    latent_dim = 64

    ############################################
    '''
    构建模型
    '''
    autoencoder = Autoencoder(latent_dim, shape)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    
    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    validation_data = (x_test, x_test))
    ############################################
    
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()
    
    # Plotting both the noisy images and the denoised images produced by the autoencoder.
    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    
    # calculate the precision of image recovery
    N = len(decoded_imgs)
    psnr_avg = 0
    for i in range(N):
        psnr_avg += psnr(x_test[i], np.squeeze(decoded_imgs[i]))
    print('Average reconstructed PSNR: %2.4f'%(psnr_avg/N))

    n = 10
    idx = np.random.randint(1e4, size=n)
    tests = x_test[idx]
    recons = decoded_imgs[idx]
    
    plt.figure(figsize=(20, 3))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(tests[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recons[i])
        pr = psnr(tests[i],np.squeeze(recons[i]))
        plt.title('recon PSNR:%2.2f'% pr)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()

# [EOF]