# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:17:02 2024

@author: 6
"""
import numpy as np
#%% 综合练习
# 创建一个10x10的随机矩阵，归一化其所有元素到0到1之间。
matrix = np.random.rand(10, 10)
# 归一化公式： (matrix - matrix.min()) / (matrix.max() - matrix.min())
normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

# 创建一个5x5的随机矩阵，找到其最大值的索引。
matrix = np.random.rand(5, 5)
# 找到最大值的索引
max_index = np.unravel_index(np.argmax(matrix), matrix.shape)

# 创建一个10x3的随机矩阵，找出每行的最大值及其索引。
matrix = np.random.rand(10, 3)

# 找出每行的最大值
row_max = np.max(matrix, axis=1)
max_index = np.argmax(matrix, axis = 1)

# 创建一个长度为20的随机数组，找出其第二大的元素。
array = np.random.rand(20)

second_largest = np.sort(array)[-2]

# 创建一个3x3的随机矩阵，将其转换为仅包含0和1的矩阵（根据某个自定义阈值）。
matrix = np.random.rand(3, 3)

# 根据阈值（如0.5）将矩阵转换为仅包含 0 和 1 的矩阵
threshold = 0.5
binary_matrix = (matrix > threshold).astype(int)

# 创建一个包含1000个元素的数组，将其中的偶数替换为-1。
array = np.random.randint(0, 1000, size=1000)

array[array % 2 == 0] = -1
# 创建一个5x5的随机矩阵，并将其中的奇数行逆序排列。
matrix = np.random.rand(5, 5)

matrix[1::2] = matrix[1::2, ::-1]

# 创建一个长度为10的数组，查找数组中连续大于0.5的元素段。
array = np.random.rand(10)

# 查找连续大于0.5的元素段
greater_than_0_5 = np.where(array > 0.5)[0]

# 检查是否连续
consecutive_segments = np.split(greater_than_0_5, 
                                np.where(np.diff(greater_than_0_5) != 1)[0] + 1)

# 使用numpy计算两个随机数组之间的欧氏距离。
array1 = np.random.rand(5)
array2 = np.random.rand(5)

euclidean_distance = np.linalg.norm(array1 - array2)

# 生成一个10x10的随机矩阵，查找其局部最大值（即比周围八个元素都大的值）。
# 方法1：用scipy包中的maximum_filter进行处理：
matrix = np.random.rand(10, 10)
from scipy.ndimage import maximum_filter
local_max = (matrix == maximum_filter(matrix, size=3))

# 方法2：纯用numpy包处理
# 在矩阵周围填充一圈很小的数值（如负无穷），这样便于处理边界元素
padded_matrix = np.pad(matrix, pad_width=1, mode='constant', constant_values=-np.inf)
# 遍历每个元素，检查它是否是周围 8 个元素的局部最大值
for i in range(1, padded_matrix.shape[0] - 1):
    for j in range(1, padded_matrix.shape[1] - 1):
        current_value = padded_matrix[i, j]
        neighbors = padded_matrix[i-1:i+2, j-1:j+2]  # 当前元素及其周围 8 个元素
        if current_value == np.max(neighbors) and np.sum(neighbors == current_value) == 1:
            local_max[i-1, j-1] = True  # 标记局部最大值

print("原始矩阵:")
print(matrix)
print("\n局部最大值矩阵（True表示局部最大值）:")
print(local_max)

