# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:23:54 2024

@author: 6
to test this optimizer, create a sample loss function to 
minimize with respect to a single variable,x. Compute its
gradient function and solve for its minimizing parameter 
value:
    L = 2x^4 + 3x^3 + 2
\frac{dL}{dx} = 8x^3 + 9x^2
\frac{dL}{dx} is 0 at x=0, which is a saddle point and 
at x = - \frac{9}{8}, which is the global minimum. Therefore, 
the loss function is optimized at x = - \frac{9}{8}.
"""

import tensorflow as tf
import matplotlib.pyplot as plt

x_vals = tf.linspace(-2, 2, 201)
x_vals = tf.cast(x_vals, tf.float32)

def loss(x):
  return 2*(x**4) + 3*(x**3) + 2

def grad(f, x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    result = f(x)
  return tape.gradient(result, x)

plt.plot(x_vals, loss(x_vals), c='k', label = "Loss function")
plt.plot(x_vals, grad(loss, x_vals), c='tab:blue', label = "Gradient function")
plt.plot(0, loss(0),  marker="o", c='g', label = "Inflection point")
plt.plot(-9/8, loss(-9/8),  marker="o", c='r', label = "Global minimum")
plt.legend()
plt.ylim(0,5)
plt.xlabel("x")
plt.ylabel("loss")
plt.title("Sample loss function and gradient");