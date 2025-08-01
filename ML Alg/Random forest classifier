import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
file_path = "/content/optimized_data.csv"
df = pd.read_csv(file_path)
num_bins = 4
labels = ["Low", "Medium-Low", "Medium-High", "High"]
df["artist_terms_weight_category"] = pd.qcut(
    df["metadata/artist_terms_weight"], q=num_bins, labels=labels
)
X = df.drop(columns=["metadata/artist_terms_weight", "artist_terms_weight_category"])
y = df["artist_terms_weight_category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
train_sizes = np.linspace(0.1, 0.8, 9).tolist()
train_scores = []
test_scores = []
for size in train_sizes:
    X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    rf_model.fit(X_partial, y_partial)
    train_scores.append(rf_model.score(X_partial, y_partial))
    test_scores.append(rf_model.score(X_test, y_test))
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores, label="Training Accuracy", marker="o")
plt.plot(train_sizes, test_scores, label="Validation Accuracy", marker="s")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.show()
train_losses = [1 - score for score in train_scores]
test_losses = [1 - score for score in test_scores]
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_losses, label="Training Loss", marker="o", color='r')
plt.plot(train_sizes, test_losses, label="Validation Loss", marker="s", color='b')
plt.xlabel("Training Set Size")
plt.ylabel("Loss (1 - Accuracy)")
plt.title("Loss Curve")
plt.legend()
plt.show()
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
