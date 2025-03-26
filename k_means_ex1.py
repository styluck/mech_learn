# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:39:58 2024

@author: lich5
"""
import numpy as np

def make_blobs(n_samples, n_features, n_blobs, centroids, cluster_std, random_state=None):
    """
    Generates `n_samples` data points, each with `n_features` dimensions, distributed around `n_blobs` clusters.
    The `centroids` and `cluster_std` specify the centers and standard deviations for each blob, respectively.

    Parameters:
    - n_samples (int): Total number of samples to generate.
    - n_features (int): Number of features for each sample.
    - n_blobs (int): Number of blobs (clusters).
    - centroids (ndarray): Centroids for each blob, shape (n_blobs, n_features).
    - cluster_std (ndarray): Standard deviations for each blob, shape (n_blobs, n_features).
    - random_state (int, optional): Seed for reproducibility.

    Returns:
    - X (ndarray): Generated samples, shape (n_samples, n_features).
    - y (ndarray): Cluster labels for each sample, shape (n_samples,).
    """
    # Set the random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize data and labels
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    # Determine the number of samples per blob
    roundn = int(np.ceil(n_samples / n_blobs))
    
    for i in range(n_blobs):
        # Generate points around the centroid of each blob
        blob = np.random.randn(roundn, n_features) * cluster_std[i] + centroids[i]
        start_idx = i * roundn
        end_idx = min((i + 1) * roundn, n_samples)
        X[start_idx:end_idx, :] = blob[:end_idx - start_idx, :]
        y[start_idx:end_idx] = i
    
    return X, y


#%%###################### main ####################### 
if __name__ == '__main__':
        
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    # from fcmeans import FCM
    
    # Parameter settings
    n_samples = 1000  # Number of samples
    n_features = 2    # Number of features
    random_state = 42  # Random seed
    
    '''请自己定义生成的随机聚类点参数'''
    n_blobs =         # Number of clusters
    centroids =    # Centroids
    cluster_std =   # Cluster std deviation
    
    
    # Generate data
    X, y = make_blobs(n_samples, n_features, n_blobs, 
                      centroids, cluster_std, random_state)
    
    # Visualize generated data
    color_map = ['r', 'g', 'b', 'c','m','k','y']
    plt.figure()
    for i in range(n_blobs):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=color_map[i], label=f'Blob {i + 1}',s =500)
    plt.title('Generated Blobs Data')
    plt.legend()
    plt.show()
    
    
    '''请调用kmeans函数进行聚类 '''
    
    
    # Visualize KMeans results
    plt.figure()
    for i in range(7):
        mask = kmeans_labels == i
        plt.scatter(X[mask, 0], X[mask, 1], c=color_map[i], label=f'KMeans Cluster {i + 1}', s=100)
    plt.scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], c='k', marker='x', s=400, label='Centroids')
    plt.title('KMeans Cluster Assignments and Centroids')
    plt.legend()
    plt.show()
    
    
# [EOF]