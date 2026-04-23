import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Fix: plt is the standard alias
from scipy.cluster import hierarchy as sch # Fix: missing dot and added standard alias

# Loading dataset
# Fix: read_csv (spelling) and use the correct variable name 'dataset'
dataset = pd.read_csv(" <filename> ")
X = dataset.iloc[:, [3, 4]].values
print(dataset.head())

# Visualizing the Dendrogram to find the optimal number of clusters
plt.figure(figsize=(10, 7))
# Fix: linkage method is "ward" (not "word"); hierarchy.linkage
linkage_matrix = sch.linkage(X, method="ward")
sch.dendrogram(linkage_matrix)

plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Training the Hierarchical Clustering model
from sklearn.cluster import AgglomerativeClustering 
# Fix: Agglomerative (spelling), n_clusters (plural), and linkage="ward"
# Note: 'metric' replaces 'affinity' in newer versions of sklearn
hc = AgglomerativeClustering(n_clusters=5, linkage="ward", metric="euclidean")
y_hc = hc.fit_predict(X)

# Visualizing the clusters (Optional but helpful)
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
