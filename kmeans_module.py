import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import random

def run_kmeans(transactions, k=2, max_iters=100, tol=1e-4):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(transactions)

    indices = random.sample(range(len(X)), k)
    centroids = X[indices]

    for _ in range(max_iters):
        clusters = {i: [] for i in range(k)}
        for x in X:
            distances = [np.linalg.norm(x - c) for c in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(x)

        new_centroids = []
        for i in range(k):
            if clusters[i]:
                new_centroids.append(np.mean(clusters[i], axis=0))
            else:
                new_centroids.append(X[random.randint(0, len(X)-1)])
        new_centroids = np.array(new_centroids)

        shift = np.sum([np.linalg.norm(new_centroids[i] - centroids[i]) for i in range(k)])
        if shift < tol:
            break
        centroids = new_centroids

    cluster_result = {f"Cluster {i+1}": len(clusters[i]) for i in clusters}
    return pd.DataFrame(list(cluster_result.items()), columns=["Cluster", "Number of Transactions"])