# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:45 2024

@author: lich5
"""
# import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

# Load MNIST dataset from OpenML
'''调用tensorflow包中的 fashion mnist 数据集'''


print(f"Data shape: {X_train.shape}")  
print(f"Labels shape: {y_train.shape}") 



'''对数据进行归一化和拉平操作'''



'''调用PCA函数，自行选择主成分个数进行主成分分析'''



# Print variance explained by the two components
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: {explained_variance}")

# Visualize the PCA-transformed data
# Use the first 10,000 samples for visualization to avoid clutter
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the first 10,000 samples
scatter = ax.scatter(X_pca[:10000, 0], X_pca[:10000, 1], X_pca[:10000, 2], c=y_train[:10000], cmap='tab10', alpha=0.6)

plt.colorbar(scatter, label='Digit Label')
ax.set_title('3D PCA of MNIST Data')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.show()


# Apply PCA for dimensionality reduction
pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X_train_flat)


# Reconstruct the data from reduced dimensions
X_reconstructed = pca.inverse_transform(X_reduced)

# Visualize original and reconstructed images
n_samples = 10  # Number of images to visualize
fig, axes = plt.subplots(2, n_samples, figsize=(15, 4))
idx = np.random.randint(0,60000, size=10)
for i in range(n_samples):
    # Original image
    axes[0, i].imshow(X_train_flat[idx[i]].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title("Original")
    
    # Reconstructed image
    axes[1, i].imshow(X_reconstructed[idx[i]].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title("Reconstructed")

plt.tight_layout()
plt.show()

# Compute Reconstruction Error
reconstruction_error = np.mean((X_train_flat - X_reconstructed) ** 2)
print(f"Reconstruction Error (MSE): {reconstruction_error:.4f}")


# [EOF]