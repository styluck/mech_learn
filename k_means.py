# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:35:51 2024

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
    n_blobs = 4       # Number of clusters
    centroids = np.array([[-1, -1], [0, 0], [1, 1], [2, 2]])  # Centroids
    cluster_std = np.array([[0.2, 0.4], [0.2, 0.4], [0.2, 0.6], [0.4, 0.4]])  # Cluster std deviation
    random_state = 12  # Random seed
    
    
    # Generate data
    # X, y = make_blobs(n_samples, n_features, n_blobs, centroids, cluster_std, random_state)
    
    X, y = make_blobs(n_samples, n_features, 1, centroids[1], cluster_std[-1], random_state)
    
    # Visualize generated data
    color_map = ['r', 'g', 'b', 'c','m','y','k']
    # plt.figure()
    # for i in range(n_blobs):
    #     mask = y == i
    #     plt.scatter(X[mask, 0], X[mask, 1], c=color_map[i], label=f'Blob {i + 1}')
    # plt.title('Generated Blobs Data')
    # plt.legend()
    # plt.show()
    
    # KMeans clustering
    k = 7
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=5, max_iter=300, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(X)
    centroids_kmeans = kmeans.cluster_centers_
    
    # Visualize KMeans results
    plt.figure()
    for i in range(k):
        mask = kmeans_labels == i
        plt.scatter(X[mask, 0], X[mask, 1], c=color_map[i], label=f'KMeans Cluster {i + 1}', s=10)
    plt.scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], c='k', marker='x', s=100, label='Centroids')
    plt.title('KMeans Cluster Assignments and Centroids')
    plt.legend()
    plt.show()
    
    
# [EOF]