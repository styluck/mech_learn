# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 07:22:59 2024

@author: 6
"""
import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats

#%% 绘制[0, 2*pi]区间的sin曲线，设置线段颜色、类型、marker类型、
# 并添加轴标题。
x = np.arange(0, 2 * np.pi, 0.25)

y = np.sin(x)

plt.figure()
plt.plot(x, y, ':ro', 
         linewidth=2, 
         markeredgecolor='k', 
         markerfacecolor=(0.49, 1, 0.63),  # 红绿蓝
         markersize=12)

plt.xlabel('X')
plt.ylabel('Y')

plt.show()
plt.close()
#%% 绘制单位圆、[0, 2*pi]区间的sin、cos曲线，设置线段颜色、类型、
# marker类型、并添加标签。
t = np.linspace(0, 2 * np.pi, 60)

x = np.cos(t)
y = np.sin(t)

plt.plot(t, x, ':', linewidth=2, label='x = cos(t)')
plt.plot(t, y, 'r-.', linewidth=3, label='y = sin(t)')
plt.plot(x, y, 'k', linewidth=2.5, label='unit circle')  

plt.axis('equal')


plt.xlabel('X')
plt.ylabel('Y')

plt.legend(loc='upper right')

plt.show()


#%% 绘制3x² + 2xy + 4y² = 5的曲线
# Ax² + Bxy + cy² + Dx + Ey + F = 0
# 3x² + 2xy + 4y² = 5可以被看作單位圓在關聯於對稱矩陣
# A' = [[A B/2], [B/2 C]]
# 的線性映射下的圖像，兩個與之對應的特徵值分別是半長軸和
# 半短軸的長度的平方的倒數。
A = np.array([[3, 1], [1, 4]])
r = 5

# Eigen decomposition of P
D, V = np.linalg.eig(A)

# Compute 半長軸 and 半短軸 axes
a = np.sqrt(r / D[0])  # Eigenvalue corresponding to first eigenvector
b = np.sqrt(r / D[1])  # Eigenvalue corresponding to second eigenvector

# Generate parametric t values
t = np.linspace(0, 2 * np.pi, 60)

# Parametric equation for the ellipse
xy = V @ np.array([a * np.cos(t), b * np.sin(t)])

plt.plot(xy[0, :], xy[1, :], 'k', linewidth=3)

plt.annotate('3x² + 2xy + 4y² = 5', 
             xy=(0.0, 0.0), 
             xytext=(0.25, 0.25), 
             arrowprops=dict(facecolor='black'),
             fontsize=15)
# xy: The point (x, y) in data coordinates where the annotation points.
# xytext: The position (x, y) where the text is placed. If not specified, the text is placed at the same location as xy.
plt.title('this is a ellipse', fontsize=18)

plt.axis([-1.5, 1.5, -1.2, 1.7])

plt.xlabel('X')
plt.ylabel('Y')

plt.show()


#%% 生成1000个服从正态分布随机样本，满足均值为10，方差为20，并绘制的直方图
x = np.random.normal(10, 20, 1000)

fig, axs = plt.subplots(1, 2)
axs[0].hist(x, bins=100)

# Label the axes
plt.xlabel('Sample data')  # Sample data
plt.ylabel('Frequency')      # Frequency
# Generate the CDF plot using the empirical cumulative distribution function

#%% 用以下数据绘制饼图
x = [10, 10, 20, 25, 35]
labels = ['A', 'B', 'C', 'D', 'E']

# Explode the last slice
explode = [0, 0, 0, 0, 0.1]
axs[1].pie(x, explode=explode, labels=labels, autopct='%1.1f%%')
axs[1].set_title('pie chart')

#%% 用以下数据绘制三维的表面图
# Generate t values and create a meshgrid
t = np.linspace(-np.pi, np.pi, 20)
X, Y = np.meshgrid(t, t)
Z = np.cos(X) * np.sin(Y)

# Create a figure for subplots
fig = plt.figure()

# 1st subplot: Mesh plot (wireframe in Matplotlib)
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_wireframe(X, Y, Z)
ax1.set_title('mesh')

# 2nd subplot: Surf plot (surface plot with transparency)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
surf = ax2.plot_surface(X, Y, Z, alpha=0.5)
ax2.set_title('surf')

# 3rd subplot: Surfl equivalent (surface plot with lighting effects)
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', rstride=1, cstride=1)
ax3.set_title('surfl')

# 4th subplot: Surfc equivalent (surface plot with contours)
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax4.contour(X, Y, Z, zdir='z', offset=-1, cmap='viridis')  # Adding contour plot at z=-1
ax4.set_title('surfc')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
# [EOF]
