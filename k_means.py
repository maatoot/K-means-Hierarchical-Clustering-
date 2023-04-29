import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_csv("cc.csv")

data.fillna(data.mean(numeric_only=True), inplace=True)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.iloc[:, 1:])

linked = linkage(data_scaled, method="ward")

plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.show()

n_clusters_hc = 3

hc = AgglomerativeClustering(n_clusters=n_clusters_hc)
hc_clusters = hc.fit_predict(data_scaled)

data["HC_Cluster"] = hc_clusters

sns.scatterplot(data=data, x="BALANCE", y="PURCHASES", hue="HC_Cluster")
plt.show()

n_clusters_kmeans = 3

kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
kmeans_clusters = kmeans.fit_predict(data_scaled)

data["KMeans_Cluster"] = kmeans_clusters

sns.scatterplot(data=data, x="BALANCE", y="PURCHASES", hue="KMeans_Cluster")
plt.show()

print("Hierarchical Clustering:")
print(data["HC_Cluster"].value_counts())
print("\nK-means Clustering:")
print(data["KMeans_Cluster"].value_counts())
