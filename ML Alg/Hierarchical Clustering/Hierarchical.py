import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

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

# Define features (X) and target (y)
X = df.drop(columns=["metadata/artist_terms_weight", "artist_terms_weight_category", "artist_terms_weight_category_encoded"])
y = df["artist_terms_weight_category_encoded"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SVM Classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Compute performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# Accuracy Curve
train_sizes = np.linspace(0.1, 0.99, 10)
train_scores = []
test_scores = []

for size in train_sizes:
    X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    svm_model.fit(X_partial, y_partial)
    train_scores.append(svm_model.score(X_partial, y_partial))
    test_scores.append(svm_model.score(X_test, y_test))

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores, label="Training Accuracy", marker='o')
plt.plot(train_sizes, test_scores, label="Testing Accuracy", marker='s')
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.show()

# Loss Curve (Using 1 - accuracy as loss approximation)
loss_train = [1 - score for score in train_scores]
loss_test = [1 - score for score in test_scores]

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, loss_train, label="Training Loss", marker='o', color='r')
plt.plot(train_sizes, loss_test, label="Testing Loss", marker='s', color='g')
plt.xlabel("Training Size")
plt.ylabel("Loss (1 - Accuracy)")
plt.title("Loss Curve")
plt.legend()
plt.show()

# Hierarchical Clustering
linkage_matrix = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Assign clusters
y_hc = fcluster(linkage_matrix, t=4, criterion='maxclust')

# Evaluate Clustering Performance
sil_score = silhouette_score(X_scaled, y_hc)
db_index = davies_bouldin_score(X_scaled, y_hc)
ch_score = calinski_harabasz_score(X_scaled, y_hc)

print(f"Silhouette Score: {sil_score:.2f}")
print(f"Davies-Bouldin Index: {db_index:.2f}")
print(f"Calinski-Harabasz Score: {ch_score:.2f}")
