# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# 2. Load Dataset
df = pd.read_csv('Mall_Customers.csv')
df_clean = df.drop(columns=['CustomerID', 'Gender'])  # Drop non-numeric columns

# 3. Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# 4. Elbow Method for KMeans (to find optimal K)
wcss = [KMeans(n_clusters=i, random_state=42).fit(X_scaled).inertia_ for i in range(1, 11)]
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()

# 5. Apply Clustering Algorithms

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.8, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=5)
df['Agglo_Cluster'] = agglo.fit_predict(X_scaled)

# 6. Visualize KMeans Clusters
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['KMeans_Cluster'], palette='Set1')
plt.title('K-Means Clustering')
plt.show()

# 7. Dendrogram for Hierarchical Clustering
linked = linkage(X_scaled, method='ward')
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# 8. Evaluate with Silhouette Score
print("KMeans Silhouette Score:", silhouette_score(X_scaled, df['KMeans_Cluster']))
print("Agglomerative Silhouette Score:", silhouette_score(X_scaled, df['Agglo_Cluster']))

# DBSCAN evaluation: Skip if all points are noise (-1)
if len(set(df['DBSCAN_Cluster'])) > 1 and -1 not in set(df['DBSCAN_Cluster']):
    print("DBSCAN Silhouette Score:", silhouette_score(X_scaled, df['DBSCAN_Cluster']))
else:
    print("DBSCAN: Not applicable")

# 9. Cluster Summary (KMeans)
print("\nKMeans Cluster Summary:")
print(df.groupby('KMeans_Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())
