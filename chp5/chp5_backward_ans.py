# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:03:03 2026

@author: lich5

课堂练习参考答案：手写反向传播 + 激活函数对比

内容：
1. 手写 Matmul / 激活函数 / Softmax / Log 的 forward 和 backward
2. 手写两层 MLP 的前向传播与反向传播
3. 用 TensorFlow 自动求导验证手写梯度
4. 在 MNIST 上训练并比较不同激活函数

说明：
- 为了数值稳定，Softmax 使用了 shift 技巧
- 交叉熵损失使用：L = - mean(sum(y * log_prob))
- 输出层默认仍使用 Softmax + Log
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =========================================================
# 1. 数据集
# =========================================================
def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return (x_train, y_train), (x_test, y_test)


def one_hot(labels, num_classes=10):
    out = np.zeros((len(labels), num_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


# =========================================================
# 2. 基础层：矩阵乘法
# =========================================================
class Matmul:
    def __init__(self):
        self.cache = {}

    def forward(self, x, W):
        """
        x: shape (N, d)
        W: shape (d, m)
        return: shape (N, m)
        """
        out = np.matmul(x, W)
        self.cache['x'] = x
        self.cache['W'] = W
        return out

    def backward(self, grad_out):
        """
        grad_out: shape (N, m)

        return:
            grad_x: shape (N, d)
            grad_W: shape (d, m)
        """
        x = self.cache['x']
        W = self.cache['W']

        grad_x = np.matmul(grad_out, W.T)
        grad_W = np.matmul(x.T, grad_out)

        return grad_x, grad_W


# =========================================================
# 3. 激活函数层
# =========================================================
class ReLU:
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        self.cache['x'] = x
        out = np.maximum(x, 0.0)
        return out

    def backward(self, grad_out):
        x = self.cache['x']
        grad_x = grad_out * (x > 0).astype(x.dtype)
        return grad_x


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.cache = {}

    def forward(self, x):
        self.cache['x'] = x
        out = np.where(x > 0, x, self.alpha * x)
        return out

    def backward(self, grad_out):
        x = self.cache['x']
        slope = np.where(x > 0, 1.0, self.alpha).astype(x.dtype)
        grad_x = grad_out * slope
        return grad_x


class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.cache = {}

    def forward(self, x):
        self.cache['x'] = x
        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1.0))
        return out

    def backward(self, grad_out):
        x = self.cache['x']
        dx = np.where(x > 0, 1.0, self.alpha * np.exp(x)).astype(x.dtype)
        grad_x = grad_out * dx
        return grad_x


class GELU:
    """
    使用 tanh 近似：
    GELU(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    """
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        self.cache['x'] = x
        c = np.sqrt(2.0 / np.pi)
        u = c * (x + 0.044715 * x**3)
        t = np.tanh(u)
        out = 0.5 * x * (1.0 + t)
        self.cache['u'] = u
        self.cache['t'] = t
        return out

    def backward(self, grad_out):
        x = self.cache['x']
        t = self.cache['t']
        c = np.sqrt(2.0 / np.pi)

        du_dx = c * (1.0 + 3.0 * 0.044715 * x**2)
        dt_dx = (1.0 - t**2) * du_dx

        dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * dt_dx
        grad_x = grad_out * dgelu_dx
        return grad_x


# =========================================================
# 4. Softmax 与 Log
# =========================================================
class Softmax:
    def __init__(self):
        self.cache = {}
        self.eps = 1e-12

    def forward(self, x):
        """
        x: shape (N, C)
        """
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shift)
        out = exp_x / (np.sum(exp_x, axis=1, keepdims=True) + self.eps)

        self.cache['out'] = out
        return out

    def backward(self, grad_out):
        """
        softmax 的 Jacobian:
            J = diag(s) - s s^T
        对每个样本：
            grad_x = J @ grad_out
                   = s * grad_out - s * sum(s * grad_out)
        """
        s = self.cache['out']  # (N, C)
        inner = np.sum(s * grad_out, axis=1, keepdims=True)   # (N, 1)
        grad_x = s * (grad_out - inner)
        return grad_x


class Log:
    def __init__(self):
        self.cache = {}
        self.eps = 1e-12

    def forward(self, x):
        self.cache['x'] = x
        out = np.log(x + self.eps)
        return out

    def backward(self, grad_out):
        x = self.cache['x']
        grad_x = grad_out / (x + self.eps)
        return grad_x


# =========================================================
# 5. 网络结构
# =========================================================
def build_activation(name):
    if name == 'relu':
        return ReLU()
    elif name == 'leaky_relu':
        return LeakyReLU(alpha=0.01)
    elif name == 'elu':
        return ELU(alpha=1.0)
    elif name == 'gelu':
        return GELU()
    else:
        raise ValueError(f'Unknown activation: {name}')


class MyMLP:
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10,
                 activation='relu', lr=1e-3, seed=42):
        np.random.seed(seed)

        self.lr = lr
        self.activation_name = activation

        # Xavier/He 风格的简单初始化
        self.W1 = (np.random.randn(input_dim + 1, hidden_dim).astype(np.float32)
                   * np.sqrt(2.0 / (input_dim + 1)))
        self.W2 = (np.random.randn(hidden_dim, output_dim).astype(np.float32)
                   * np.sqrt(2.0 / hidden_dim))

        self.fc1 = Matmul()
        self.act1 = build_activation(activation)
        self.fc2 = Matmul()
        self.softmax = Softmax()
        self.log = Log()

        self.cache = {}

    def add_bias(self, x):
        bias = np.ones((x.shape[0], 1), dtype=np.float32)
        return np.concatenate([x, bias], axis=1)

    def forward(self, x):
        """
        x: shape (N, 28, 28) or (N, 784)
        """
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        x = self.add_bias(x)

        z1 = self.fc1.forward(x, self.W1)
        h1 = self.act1.forward(z1)
        z2 = self.fc2.forward(h1, self.W2)
        prob = self.softmax.forward(z2)
        log_prob = self.log.forward(prob)

        self.cache['x'] = x
        self.cache['z1'] = z1
        self.cache['h1'] = h1
        self.cache['z2'] = z2
        self.cache['prob'] = prob
        self.cache['log_prob'] = log_prob

        return log_prob

    def backward(self, labels):
        """
        labels: one-hot, shape (N, C)
        """
        N = labels.shape[0]
        log_prob = self.cache['log_prob']

        # L = - mean(sum(y * log_prob))
        grad_log_prob = -labels / N

        grad_prob = self.log.backward(grad_log_prob)
        grad_z2 = self.softmax.backward(grad_prob)
        grad_h1, grad_W2 = self.fc2.backward(grad_z2)
        grad_z1 = self.act1.backward(grad_h1)
        grad_x, grad_W1 = self.fc1.backward(grad_z1)

        self.grad_W1 = grad_W1
        self.grad_W2 = grad_W2

    def update(self):
        self.W1 -= self.lr * self.grad_W1
        self.W2 -= self.lr * self.grad_W2

    def compute_loss(self, log_prob, labels):
        loss = -np.mean(np.sum(labels * log_prob, axis=1))
        return loss

    def compute_accuracy(self, log_prob, labels):
        pred = np.argmax(log_prob, axis=1)
        truth = np.argmax(labels, axis=1)
        return np.mean(pred == truth)

    def train_one_step(self, x, y):
        log_prob = self.forward(x)
        self.backward(y)
        self.update()

        loss = self.compute_loss(log_prob, y)
        acc = self.compute_accuracy(log_prob, y)
        return loss, acc

    def predict(self, x):
        log_prob = self.forward(x)
        return np.argmax(log_prob, axis=1)


# =========================================================
# 6. 随机小样本测试：用于验证梯度
# =========================================================
def manual_gradient_demo():
    np.random.seed(0)
    tf.random.set_seed(0)

    N, d, h, c = 4, 6, 5, 3
    x = np.random.randn(N, d).astype(np.float32)
    labels = np.zeros((N, c), dtype=np.float32)
    labels[np.arange(N), np.random.randint(0, c, size=N)] = 1.0

    # 网络参数
    W1_np = np.random.randn(d, h).astype(np.float32)
    W2_np = np.random.randn(h, c).astype(np.float32)

    # ========== 前向 ==========
    fc1 = Matmul()
    relu = ReLU()
    fc2 = Matmul()
    softmax = Softmax()
    log = Log()

    h1 = fc1.forward(x, W1_np)
    h1_relu = relu.forward(h1)
    logits = fc2.forward(h1_relu, W2_np)
    prob = softmax.forward(logits)
    log_prob = log.forward(prob)

    loss_manual = -np.mean(np.sum(labels * log_prob, axis=1))

    # ========== 反向 ==========
    grad_log_prob = -labels / N
    grad_prob = log.backward(grad_log_prob)
    grad_logits = softmax.backward(grad_prob)
    grad_h1, grad_W2_manual = fc2.backward(grad_logits)
    grad_h1_pre = relu.backward(grad_h1)
    grad_x, grad_W1_manual = fc1.backward(grad_h1_pre)

    # ========== TensorFlow ==========
    W1_tf = tf.Variable(W1_np)
    W2_tf = tf.Variable(W2_np)
    x_tf = tf.constant(x)
    y_tf = tf.constant(labels)

    with tf.GradientTape() as tape:
        h1_tf = tf.matmul(x_tf, W1_tf)
        h1_relu_tf = tf.nn.relu(h1_tf)
        logits_tf = tf.matmul(h1_relu_tf, W2_tf)
        prob_tf = tf.nn.softmax(logits_tf, axis=1)
        log_prob_tf = tf.math.log(prob_tf + 1e-12)
        loss_tf = -tf.reduce_mean(tf.reduce_sum(y_tf * log_prob_tf, axis=1))

    grad_W1_tf, grad_W2_tf = tape.gradient(loss_tf, [W1_tf, W2_tf])

    grad_W1_tf = grad_W1_tf.numpy()
    grad_W2_tf = grad_W2_tf.numpy()

    # ========== 误差 ==========
    err_W1 = np.linalg.norm(grad_W1_manual - grad_W1_tf)
    err_W2 = np.linalg.norm(grad_W2_manual - grad_W2_tf)

    print("=== 手写梯度 vs TensorFlow 自动求导 ===")
    print(f"manual loss = {loss_manual:.8f}")
    print(f"tf loss     = {loss_tf.numpy():.8f}")
    print(f"||grad_W1_manual - grad_W1_tf|| = {err_W1:.8e}")
    print(f"||grad_W2_manual - grad_W2_tf|| = {err_W2:.8e}")

    return {
        'loss_manual': loss_manual,
        'loss_tf': float(loss_tf.numpy()),
        'err_W1': err_W1,
        'err_W2': err_W2
    }


# =========================================================
# 7. TensorFlow 梯度对比参考框架
# =========================================================
def tensorflow_gradient_check():
    np.random.seed(0)
    tf.random.set_seed(0)

    N, d, h, c = 4, 6, 5, 3
    x = np.random.randn(N, d).astype(np.float32)
    labels = np.zeros((N, c), dtype=np.float32)
    labels[np.arange(N), np.random.randint(0, c, size=N)] = 1.0

    W1 = tf.Variable(np.random.randn(d, h).astype(np.float32))
    W2 = tf.Variable(np.random.randn(h, c).astype(np.float32))

    with tf.GradientTape() as tape:
        h1 = tf.matmul(x, W1)
        h1 = tf.nn.relu(h1)
        logits = tf.matmul(h1, W2)
        prob = tf.nn.softmax(logits, axis=1)
        log_prob = tf.math.log(prob + 1e-12)
        loss = -tf.reduce_mean(tf.reduce_sum(labels * log_prob, axis=1))

    grad_W1, grad_W2 = tape.gradient(loss, [W1, W2])

    print("TensorFlow grad_W1 shape:", grad_W1.shape)
    print("TensorFlow grad_W2 shape:", grad_W2.shape)

    return grad_W1.numpy(), grad_W2.numpy()


# =========================================================
# 8. MNIST 训练实验
# =========================================================
def run_experiment(activation='relu', epochs=5, batch_size=256, lr=1e-3):
    (x_train, y_train), (x_test, y_test) = load_mnist_dataset()

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    y_train_oh = one_hot(y_train, 10)
    y_test_oh = one_hot(y_test, 10)

    model = MyMLP(
        input_dim=784,
        hidden_dim=100,
        output_dim=10,
        activation=activation,
        lr=lr,
        seed=42
    )

    n_train = x_train.shape[0]

    for epoch in range(epochs):
        idx = np.random.permutation(n_train)
        x_train = x_train[idx]
        y_train_oh = y_train_oh[idx]

        epoch_losses = []
        epoch_accs = []

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            xb = x_train[start:end]
            yb = y_train_oh[start:end]

            loss, acc = model.train_one_step(xb, yb)
            epoch_losses.append(loss)
            epoch_accs.append(acc)

        test_log_prob = model.forward(x_test)
        test_loss = model.compute_loss(test_log_prob, y_test_oh)
        test_acc = model.compute_accuracy(test_log_prob, y_test_oh)

        print(f"[{activation}] epoch {epoch+1:02d} | "
              f"train loss {np.mean(epoch_losses):.4f} | "
              f"train acc {np.mean(epoch_accs):.4f} | "
              f"test loss {test_loss:.4f} | "
              f"test acc {test_acc:.4f}")

    return model, float(test_acc)


# =========================================================
# 9. 不同激活函数比较
# =========================================================
def compare_activations():
    activations = ['relu', 'leaky_relu', 'elu', 'gelu']
    results = []

    for act in activations:
        print("\n" + "=" * 60)
        print(f"Running experiment with activation = {act}")
        print("=" * 60)

        _, test_acc = run_experiment(
            activation=act,
            epochs=3,
            batch_size=256,
            lr=1e-3
        )
        results.append((act, test_acc))

    print("\n最终结果：")
    for act, metric in results:
        print(f"{act:12s} test_acc = {metric:.4f}")

    return results


# =========================================================
# 10. 主程序
# =========================================================
if __name__ == '__main__':
    # Part A: 随机小样本梯度检查
    manual_gradient_demo()

    # Part B: 查看 TensorFlow 自动求导结果
    tensorflow_gradient_check()

    # Part C: 在 MNIST 上训练单个激活函数
    run_experiment(activation='relu', epochs=5, batch_size=256, lr=1e-3)

    # Part D: 比较不同激活函数
    compare_activations()
