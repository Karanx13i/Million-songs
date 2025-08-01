import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load the dataset
file_path = "/content/optimized_data.csv"
df = pd.read_csv(file_path)

# Define bins and labels for categorization
num_bins = 4  # Dividing into 4 categories
labels = ["Low", "Medium-Low", "Medium-High", "High"]

df["artist_terms_weight_category"] = pd.qcut(
    df["metadata/artist_terms_weight"], q=num_bins, labels=labels
)

# Encode categorical labels into numerical values
label_encoder = LabelEncoder()
df["artist_terms_weight_category_encoded"] = label_encoder.fit_transform(df["artist_terms_weight_category"])

# Define features (X)
X = df.drop(columns=["metadata/artist_terms_weight", "artist_terms_weight_category", "artist_terms_weight_category_encoded"])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Compute performance metrics
silhouette_avg = silhouette_score(X_scaled, df["cluster"])
davies_bouldin = davies_bouldin_score(X_scaled, df["cluster"])
calinski_harabasz = calinski_harabasz_score(X_scaled, df["cluster"])

print(f"Silhouette Score: {silhouette_avg:.2f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.2f}")

# Compute classification metrics
precision = precision_score(df["artist_terms_weight_category_encoded"], df["cluster"], average='weighted', zero_division=1)
recall = recall_score(df["artist_terms_weight_category_encoded"], df["cluster"], average='weighted')
f1 = f1_score(df["artist_terms_weight_category_encoded"], df["cluster"], average='weighted')
accuracy = accuracy_score(df["artist_terms_weight_category_encoded"], df["cluster"])

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# Visualizing Clusters (Using first two principal components for visualization)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df["cluster"], palette="viridis")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering Visualization")
plt.legend(title="Cluster")
plt.show()

# Accuracy Curve (Within-Cluster Sum of Squares vs. Number of Clusters)
inertia = []
k_values = range(1, 10)
for k in k_values:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertia.append(kmeans_temp.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Optimal Clusters")
plt.show()

# Loss Curve (Inverse of Silhouette Score)
silhouette_scores = []
for k in range(2, 10):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans_temp.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), [1 - score for score in silhouette_scores], marker='s', color='r')
plt.xlabel("Number of Clusters")
plt.ylabel("Loss (1 - Silhouette Score)")
plt.title("Loss Curve")
plt.show()

# Confusion Matrix (Comparing Cluster Assignments to Original Labels)
conf_matrix = confusion_matrix(df["artist_terms_weight_category_encoded"], df["cluster"])
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Cluster")
plt.ylabel("Actual Category")
plt.title("Confusion Matrix")
plt.show()

# Print cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)
