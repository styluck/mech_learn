# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:40:48 2024

@author: lich5
演示TensorFlow的自动微分功能（计算一阶和二阶导数）、矩阵运算的梯度计算以及张量均值计算。
"""
import tensorflow as tf
import numpy as np

# %% 一阶导数计算: y=x^2 在 x=3 处的导数。
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2
    dy_dx = tape.gradient(y, x)
    
print(dy_dx.numpy()) # 输出结果：6.0

# %% 矩阵运算的梯度计算：计算线性变换 y=x*w+b 的均值损失对参数 w,b,x 的梯度。
w = tf.Variable(np.arange(6).reshape(3,2).astype('f'), name='w')
b = tf.Variable(np.arange(2).astype('f'), name='b')
x = tf.Variable(np.arange(1,4).reshape(1,3).astype('f'), name='x')

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    #tf.reduce_mean()函数用于计算张量tensor沿着指定的数轴（tensor中的某一维度）上的平均值，主要用于降维或者计算tensor（图像）的平均值。
    # 这里计算均值：(4+5)/2=4.5
    loss = tf.reduce_mean(y)

[dl_dw, dl_db, dl_dx] = tape.gradient(loss, [w, b, x])
"""
也可以传递为如下形式:
my_vars = {'w': w, 'b': b}
tape.gradient(loss, my_vars)
"""
print("dl_dw:", dl_dw.numpy())
print("dl_db:", dl_db.numpy())


# %% 二阶导数计算：计算 y=x^2 的二阶导数（即曲率）。

with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
        gg.watch(x)
        y = x * x
    dy_dx = gg.gradient(y, x)     # 求一阶导数
d2y_dx2 = g.gradient(dy_dx, x)    # 求二阶导数

print("d2y_dx2:", d2y_dx2.numpy())

# %%张量均值计算：使用 tf.reduce_mean 计算不同维度的均值。
x = [[1,2,3],[1,1,3]]
xx = tf.cast(x,tf.float32)  
#tf.cast(input_data,dtype,name=None)执行的是张量的数据类型转换
mean_all = tf.reduce_mean(xx)
mean_0 = tf.reduce_mean(xx,axis=0)  #axis=0计算列方向
mean_1 = tf.reduce_mean(xx,axis=1)  #axis=1计算行方向

print("mean_all:", mean_all.numpy())
print("mean_0:", mean_0.numpy())
print("mean_1:", mean_1.numpy())

# [EOF]
