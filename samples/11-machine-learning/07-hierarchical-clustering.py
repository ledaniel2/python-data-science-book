import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.5)
model = AgglomerativeClustering(n_clusters=4)
y_pred = model.fit_predict(X)
#plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.title('Hierarchical Clustering')
#plt.show()
linked = linkage(X, method='ward')
dendrogram(linked)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()
