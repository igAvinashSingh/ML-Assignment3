import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("kmeans - kmeans_blobs.csv")


# Normalize the dataset using min-max scaling
def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())


df_normalized = normalize_data(df)


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def k_means(data, k, max_iters=100, tol=1e-4):
    np.random.seed(42)
    centroids = data[np.random.choice(len(data), k, replace=False)]
    prev_centroids = np.zeros_like(centroids)
    clusters = np.zeros(len(data))

    for _ in range(max_iters):
        for i, point in enumerate(data):
            distances = euclidean_distance(point, centroids)
            clusters[i] = np.argmin(distances)

        prev_centroids = centroids.copy()
        for j in range(k):
            cluster_points = data[clusters == j]
            if len(cluster_points) > 0:
                centroids[j] = np.mean(cluster_points, axis=0)

        if np.linalg.norm(centroids - prev_centroids) < tol:
            break

    return clusters, centroids


# Convert dataframe to numpy array
data_array = df_normalized.to_numpy()

# Run k-means for k=2 and k=3
clusters_k2, centroids_k2 = k_means(data_array, k=2)
clusters_k3, centroids_k3 = k_means(data_array, k=3)


def plot_clusters(data, clusters, centroids, k):
    plt.figure(figsize=(8, 6))
    for i in range(k):
        plt.scatter(data[clusters == i, 0], data[clusters == i, 1], label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
    plt.xlabel('x1 (Normalized)')
    plt.ylabel('x2 (Normalized)')
    plt.title(f'K-Means Clustering (k={k})')
    plt.legend()
    plt.show()


# Plot results for k=2 and k=3
plot_clusters(data_array, clusters_k2, centroids_k2, k=2)
plot_clusters(data_array, clusters_k3, centroids_k3, k=3)