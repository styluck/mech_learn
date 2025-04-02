# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:52:07 2024

@author: 6
"""
import numpy as np
import matplotlib.pyplot as plt
#%% 创建一个包含两个子图的图形，
# 在第一个子图中绘制 y = sin(x)。
# 在第二个子图中绘制 y = cos(x)。
# 添加轴标签并给每个子图添加标题。
x = np.linspace(0, 2*np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

fig, axs = plt.subplots(2)
axs[0].plot(x, y_sin)
axs[0].set_title('Sine Curve')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

axs[1].plot(x, y_cos)
axs[1].set_title('Cosine Curve')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

plt.show()


#%% 生成以下数据的饼图：
# 数据：[15, 30, 45, 10]
# 标签：['apple', 'banana', 'cherry', 'date']
# 将 "cherry" 切片爆炸 0.1。
# 在饼图中标注每个切片的百分比。
data = [15, 30, 45, 10]
labels = ['apple', 'banana', 'cherry', 'date']
colors = ['red', 'yellow', 'green', 'purple']
explode = (0, 0, 0.1, 0)

fig, ax = plt.subplots()
ax.pie(data, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.show()

#%% 创建函数 z = sin(sqrt(x^2 + y^2)) 的 3D 表面图。
x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
z = np.sin(np.sqrt(x**2 + y**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Surface Plot of z = sin(sqrt(x^2 + y^2))')

plt.show()


#%% *创建一个 3D 箭头图，其中向量从原点出发并指向不同方向
# 使用 cos 和 sin 函数计算相应的 u 和 v 分量。
# 在图表中添加标签和标题。
angles = np.linspace(-90, 90, 10)
angles_rad = np.deg2rad(angles)
u = np.cos(angles_rad)
v = np.sin(angles_rad)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, u, v, np.ones_like(u), length=1, color='b')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Quiver Plot')

plt.show()

#%% 创建一个参数螺旋的 3D 图，使用以下参数：
# x = 10 * cos(t), y = 10 * sin(t), z = t，
# 其中 t 是从 0 到 4π 的值范围，包含 100 个点。
# 添加轴标签和标题。
t = np.linspace(0, 4*np.pi, 100)
x_helix = 10 * np.cos(t)
y_helix = 10 * np.sin(t)
z_helix = t

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_helix, y_helix, z_helix)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Parametric Helix')

plt.show()