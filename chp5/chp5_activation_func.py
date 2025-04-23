# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:01:32 2024

@author: lich5
练习：
1. 将第一层的ReLU激活函数修改为GeLU、ELU、LeakyReLU进行测试
2. 将第二次的softmax激活函数修改为tanh进行测试
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    #normalize
    x = x/255.0
    x_test = x_test/255.0
    
    return (x, y), (x_test, y_test)


class Matmul:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x, W):
        h = np.matmul(x, W)
        self.mem={'x': x, 'W':W}
        return h
    
    def backward(self, grad_y):
        '''
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        '''
        x = self.mem['x']
        W = self.mem['W']
        
        ####################
        '''计算矩阵乘法的对应的梯度'''
        grad_W = np.matmul(x.T,grad_y)
        grad_x = np.matmul(grad_y,W.T)
        ####################
        return grad_x, grad_W


class Relu:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x):
        self.mem['x']=x
        return np.where(x > 0, x, np.zeros_like(x))
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        ####################
        '''计算relu 激活函数对应的梯度'''
        grad_x = grad_y > 0
        ####################
        return grad_x
    


class Softmax:
    '''
    softmax over last dimention
    '''
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}
        
    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp/(partition+self.epsilon)
        
        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        s = self.mem['out']
        sisj = np.matmul(np.expand_dims(s,axis=2), np.expand_dims(s, axis=1)) # (N, c, c)
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj) #(N, 1, c)
        tmp = np.squeeze(tmp, axis=1)
        tmp = -tmp+grad_y*s 
        return tmp
    
class Log:
    '''
    softmax over last dimention
    '''
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}
        
    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        out = np.log(x+self.epsilon)
        
        self.mem['x'] = x
        return out
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        x = self.mem['x']
        
        return 1./(x+1e-12) * grad_y
    
    
class myModel:
    def __init__(self):
        
        self.W1 = np.random.normal(size=[28*28+1, 100])
        self.W2 = np.random.normal(size=[100, 10])
        
        self.mul_h1 = Matmul()
        self.mul_h2 = Matmul()
        self.relu = Relu()
        self.softmax = Softmax()
        self.log = Log()
        
        
    def forward(self, x):
        x = x.reshape(-1, 28*28)
        bias = np.ones(shape=[x.shape[0], 1])
        x = np.concatenate([x, bias], axis=1)
        
        self.h1 = self.mul_h1.forward(x, self.W1) # shape(5, 4)
        self.h1_relu = self.relu.forward(self.h1)
        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)
        self.h2_soft = self.softmax.forward(self.h2)
        self.h2_log = self.log.forward(self.h2_soft)
            
    def backward(self, label):
        self.h2_log_grad = self.log.backward(-label)
        self.h2_soft_grad = self.softmax.backward(self.h2_log_grad)
        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_soft_grad)
        self.h1_relu_grad = self.relu.backward(self.h2_grad)
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_relu_grad)
        
    def predict(self, x):
        x = x.reshape(-1, 28*28)
        bias = np.ones(shape=[x.shape[0], 1])
        x = np.concatenate([x, bias], axis=1)
        
        h1 = self.mul_h1.forward(x, self.W1) # shape(5, 4)
        h1_relu = self.relu.forward(h1)
        h2 = self.mul_h2.forward(h1_relu, self.W2)
        h2_soft = self.softmax.forward(h2)
        h2_log = self.log.forward(h2_soft)
        
        return np.argmax(h2_log, axis = 1)
    
    def train_one_step(self, x, y):
        self.forward(x)
        self.backward(y)
        self.W1 -= 1e-4* self.W1_grad
        self.W2 -= 1e-4* self.W2_grad
        loss = self.compute_loss(self.h2_log, y)
        accuracy = self.compute_accuracy(self.h2_log, y)
        return loss, accuracy
    
    
    def compute_loss(self, log_prob, labels):
         return np.mean(np.sum(-log_prob*labels, axis=1))


    def compute_accuracy(self, log_prob, labels):
        predictions = np.argmax(log_prob, axis=1)
        truth = np.argmax(labels, axis=1)
        return np.mean(predictions==truth)


    def test(self, x, y):
        self.forward(x)
        loss = self.compute_loss(self.h2_log, y)
        accuracy = self.compute_accuracy(self.h2_log, y)
        return loss, accuracy


if __name__ == '__main__':
    
    
    ##########################################################
    '''
        测试随机算例
    '''
    x = np.random.normal(size=[5, 6], scale=5.0, loc=1)
    label = np.zeros_like(x)
    label[0, 1]=1.
    label[1, 0]=1
    label[2, 3]=1
    label[3, 5]=1
    label[4, 0]=1
    
    x = np.random.normal(size=[5, 6])
    W1 = np.random.normal(size=[6, 5])
    W2 = np.random.normal(size=[5, 6])
    
    mul_h1 = Matmul()
    mul_h2 = Matmul()
    relu = Relu()
    softmax = Softmax()
    log = Log()
    
    h1 = mul_h1.forward(x, W1) # shape(5, 4)
    h1_relu = relu.forward(h1)
    h2 = mul_h2.forward(h1_relu, W2)
    h2_soft = softmax.forward(h2)
    h2_log = log.forward(h2_soft)
    
    h2_log_grad = log.backward(label)
    h2_soft_grad = softmax.backward(h2_log_grad)
    h2_grad, W2_grad = mul_h2.backward(h2_soft_grad)
    h1_relu_grad = relu.backward(h2_grad)
    h1_grad, W1_grad = mul_h1.backward(h1_relu_grad)
    
    print(h2_log_grad)
    print('--'*20)
    # print(W2_grad)
    
    with tf.GradientTape() as tape:
        x, W1, W2, label = tf.constant(x), tf.constant(W1), tf.constant(W2), tf.constant(label)
        tape.watch(W1)
        tape.watch(W2)
        h1 = tf.matmul(x, W1)
        h1_relu = tf.nn.relu(h1)
        h2 = tf.matmul(h1_relu, W2)
        prob = tf.nn.softmax(h2)
        log_prob = tf.math.log(prob)
        loss = tf.reduce_sum(label * log_prob)
        grads = tape.gradient(loss, [prob])
        print (grads[0].numpy())
    
    ##########################################################
    '''
        利用mnist数据集算例
    '''
    
    import matplotlib.pyplot as plt


    # 创建模型
    (x_train, y_train), (x_test, y_test) = mnist_dataset()
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)

    train_label = np.zeros(shape=[x_train.shape[0], 10])
    test_label = np.zeros(shape=[x_test.shape[0], 10])
    train_label[np.arange(x_train.shape[0]), np.array(y_train)] = 1.
    test_label[np.arange(x_test.shape[0]), np.array(y_test)] = 1.
    
    model = myModel()
    
    for epoch in range(10):
        loss, accuracy = model.train_one_step(x_train, train_label)
        print('epoch', epoch, ': loss', loss, '; accuracy', accuracy)
    loss, accuracy = model.test(x_test, test_label)
    
    print('test loss', loss, '; accuracy', accuracy)

    
    # # 加载模型
    # model = tf.keras.models.load_model('my_model.h5')
    idx = np.random.randint(1e4,size=9)
    images = x_test[idx,:]
    y_ = y_test[idx]
    # 测试模型
    def plot_mnist_3_3(images, y_, y=None):
        assert images.shape[0] == len(y_)
        fig, axes = plt.subplots(3, 3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].reshape([28,28]), cmap='binary')
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
    
    plot_mnist_3_3(images, y_, predictions[idx])

# [EOF]
