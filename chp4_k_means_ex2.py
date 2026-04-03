# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:41:27 2024

@author: lich5
"""

import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt


# %%   基于K均值聚类的图像分割——灰度图像
# ******************************* 读取图像数据 ********************************
# Load grayscale image
x = Image.open('coins.png').convert('L')
x = np.array(x)  # Convert to numpy array
print(f"Image shape: {x.shape}, dtype: {x.dtype}")

y = x.flatten().astype(np.float64)  # Flatten the image and convert to double

# ************************* 调用kmeans函数进行聚类分割 *************************
startdata = np.array([[0], [150]])  # Initial cluster centers
kmeans = KMeans(n_clusters=2, init=startdata, n_init=1)
idpixel = kmeans.fit_predict(y.reshape(-1, 1))  # Perform clustering
idbw = (idpixel == 1)  # Binary mask for the second cluster
result = idbw.reshape(x.shape)  # Reshape back to original image size

# Display the segmented image
plt.figure()
plt.imshow(result, cmap='gray')
plt.title("Segmented Image (Grayscale)")
plt.axis('off')
plt.show()

# %%    基于K均值聚类的图像分割——真彩图像
# ******************************* 读取图像数据 ********************************
# Load color image
Duck0 = Image.open('littleduck.jpg')
Duck0 = np.array(Duck0)  # Convert to numpy array
print(f"Original Duck image shape: {Duck0.shape}, dtype: {Duck0.dtype}")

m, n, k = Duck0.shape
Duck1 = Duck0.reshape(-1, k).astype(np.float64)  # Reshape into 2D array for clustering
print(f"Reshaped Duck image shape: {Duck1.shape}, dtype: {Duck1.dtype}")

'''调用kmeans函数进行聚类分割 '''

# Display the segmented color image
plt.figure()
plt.imshow(Duck2.astype(np.uint8))
plt.title("Segmented Image (Color)")
plt.axis('off')
plt.show()

# [EOF]
  