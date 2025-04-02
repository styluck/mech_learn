# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:11:11 2024

@author: lich5
"""
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

# Load MNIST dataset from OpenML
(X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
# mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Extract data and labels
# X_train = mnist.data  # Pixel values (70,000 samples x 784 features)
# y_train = mnist.target.astype(np.uint8)  # Labels (digits 0-9)


print(f"Data shape: {X_train.shape}")  
print(f"Labels shape: {y_train.shape}") 


# Normalize the data to [0, 1] and flatten the data
X_train = X_train / 255.0  # Normalize pixel values to range [0, 1]
X_train_flat = X_train.reshape(-1, 28 * 28)  # Flatten to (num_samples, 784)

# Apply PCA to reduce to 3 components for visualization
pca = PCA(n_components=3)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_train_flat)

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
n_components = 50  # Number of principal components
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_train_flat)

# Reconstruct the data from reduced dimensions
X_reconstructed = pca.inverse_transform(X_reduced)

# Visualize original and reconstructed images
n_samples = 10  # Number of images to visualize
fig, axes = plt.subplots(2, n_samples, figsize=(15, 4))
idx = np.random.randint(0,70000, size=10)
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
