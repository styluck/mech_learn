# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:11:56 2024

@author: lich5
"""
import numpy as np
#%% 数组的创建与基本操作
# 创建一个长度为10的全零数组，并将第五个元素设为1。
array = np.zeros(10)
array[4] = 1
# 创建一个包含0到9的数组。
array = np.arange(10)
# 创建一个3x3的单位矩阵。
array = np.eye(3)
# 创建一个3x3x3的随机数组。
array = np.random.rand(3,3,3)
array = np.random.randn(3,3,3)
# 创建一个10到50的数组，并反转数组顺序。
array = np.arange(10,51)
array = array[::-1]

#%% 数组索引与切片
# 创建一个3x3矩阵，并提取其中的非零元素。
array = np.random.randn(3,3)
idx = np.where(array !=0)[0]
# 创建一个5x5的随机矩阵，并将矩阵的最大值设为1，最小值设为0。
array = np.random.rand(5,5)
array_max = np.max(array)
array_min = np.min(array)
array = np.where(array == array_max, 1, array)
array = np.where(array == array_min, 0, array)
print('the max:',np.max(array))
print('the min:',np.min(array))
# 创建一个0到99的10x10的矩阵，并提取其边界值。
array = np.arange(100).reshape(10, 10)
top_row = array[0, :]            # 第一行
bottom_row = array[-1, :]        # 最后一行
left_column = array[:, 0]        # 第一列
right_column = array[:, -1]      # 最后一列
boundary_values = np.concatenate((top_row, 
                                  bottom_row, 
                                  left_column[1:-1], 
                                  right_column[1:-1]))
print("10x10矩阵:",array)
print("\n边界值:",boundary_values)

# 创建一个长度为10的随机数组，并将其元素按升序排序。
array = np.random.rand(10)
array_sorted = np.sort(array)
print("原始随机数组:")
print(array)
print("\n升序排序后的数组:")
print(array_sorted)

#%% 统计计算

# 设置随机种子并生成一个长度为5的随机数组
np.random.seed(42)
x = np.random.rand(5)

# 创建一个100个随机数的数组，计算其平均值、中位数、方差、标准差。
array = np.random.rand(100)
mean_value = np.mean(array)      # 计算平均值
median_value = np.median(array)  # 计算中位数
variance = np.var(array)         # 计算方差
std_dev = np.std(array)          # 计算标准差

# 创建一个100个随机数的数组，计算其前5个元素的累积和。
array = np.random.rand(100)
cumulative_sum = np.cumsum(array[:5])  # 计算前5个元素的累积和

# 创建一个10x10的随机矩阵，计算其列的最大值与最小值。
matrix_10x10 = np.random.rand(10, 10)
col_max = np.max(matrix_10x10, axis=0)  # 计算每列的最大值
col_min = np.min(matrix_10x10, axis=0)  # 计算每列的最小值

# 创建一个长度为50的随机数组，统计其中大于0的元素个数。
array = np.random.randn(50)  # 生成具有正态分布的随机数组（可能包含负值）
count_positive = np.sum(array > 0)  # 统计大于0的元素个数

# 创建一个10x10的随机矩阵，统计其中大于0.5的元素个数。
matrix_10x10 = np.random.rand(10, 10)
count_greater_than_0_5 = np.sum(matrix_10x10 > 0.5)  # 统计大于0.5的元素个数

# 使用numpy生成一个1000x1000的随机矩阵，并计算其行均值和列均值。
matrix_1000x1000 = np.random.randn(1000,1000)
row_mean = np.mean(matrix_1000x1000,axis = 0)
row_mean = np.mean(matrix_1000x1000,axis = 1)


# 创建一个包含1-100数字的1000个随机整数的数组，计算每个数值的累计出现次数。
matrix_int = np.random.randint(0,101, size = 1e3)
unique, counts = np.unique(matrix_int, return_counts=True)

#%% 数学操作与线性代数
# 创建两个随机数组，分别计算它们的和、差、积、商。
array1 = np.random.rand(5)
array2 = np.random.rand(5)

array_sum = array1 + array2      # 计算和
array_diff = array1 - array2     # 计算差
array_prod = array1 * array2     # 计算积
array_div = array1 / array2      # 计算商

# 生成两个1000x100的随机矩阵，计算它们的矩阵乘法。
matrix1 = np.random.randn(1000,100)
matrix2 = np.random.randn(1000,100)
prod = matrix1.T @ matrix2

# 创建一个5x5的随机矩阵，并计算其行列式。
matrix_5x5 = np.random.rand(5, 5)
determinant = np.linalg.det(matrix_5x5) 

# 创建一个3x3的随机矩阵，并计算其逆矩阵。
matrix_3x3 = np.random.rand(3, 3)
inverse_matrix = np.linalg.inv(matrix_3x3)  # 计算逆矩阵
# 创建一个长度为5的随机数组，并计算其平方和。
array = np.random.rand(5)
sum_of_squares = np.sum(array**2)  # 计算平方和
# 创建两个随机向量，计算它们的点积。
vector1 = np.random.rand(3)
vector2 = np.random.rand(3)
dot_product = np.dot(vector1, vector2)  # 计算点积

# 创建一个20x20的随机矩阵，并计算其奇异值分解（SVD）。
matrix_20x20 = np.random.randn(20,20)
u, s, v = np.linalg.svd(matrix_20x20)

#%% 广播与形状操作
# 创建一个形状为 (3, 1) 和 (1, 3) 的两个数组，并使用广播机制计算它们的和。
array_1 = np.array([[1], [2], [3]])
array_2 = np.array([[4, 5, 6]])
result = array_1 + array_2

# 创建一个随机数组，将其形状重塑为 (2, -1)。
array = np.random.rand(6)
reshaped_array = array.reshape(2, -1)#解释：-1 表示根据数组的长度自动计算该维度的大小。

# 创建一个4x4的随机矩阵，旋转90度后，再进行转置
matrix = np.random.rand(4, 4)
rotated_matrix = np.rot90(matrix)
trans_matrix = rotated_matrix.T

# 创建一个随机数组，水平和垂直堆叠该数组的副本。
array = np.random.rand(3)
horizontal_stack = np.hstack((array, array))
vertical_stack = np.vstack((array, array))

# 创建一个长度为10的随机数组，并添加一个新的维度，使其形状变为 (10, 1)。
array = np.random.rand(10)
reshaped_array = array[:, np.newaxis]

#%% 特殊数组生成
# 创建一个5x5的棋盘矩阵（即0和1交替排列的矩阵）。
chessboard = np.zeros((5, 5), dtype=int)
chessboard[1::2, ::2] = 1  # 奇数行，偶数列设为1
chessboard[::2, 1::2] = 1  # 偶数行，奇数列设为1

# 创建一个长度为10的数组，元素值从0到1均匀分布。
array = np.linspace(0, 1, 10)  # 生成从0到1均匀分布的10个元素

# 创建一个3x3的随机矩阵，并将其对角线元素设为1。
matrix_3x3 = np.random.rand(3, 3)
np.fill_diagonal(matrix_3x3, 1)  # 将对角线元素设为1

# 创建一个3x3的随机矩阵，并将其下三角元素设为0。
matrix_3x3 = np.random.rand(3, 3)
matrix_3x3 = np.triu(matrix_3x3)  # 保留上三角部分，下三角设为0
# 创建一个4x4的随机矩阵，并将其上三角元素设为1。
matrix_4x4 = np.random.rand(4, 4)
matrix_4x4 = np.tril(matrix_4x4, -1) + np.triu(np.ones((4, 4)), 0) 

