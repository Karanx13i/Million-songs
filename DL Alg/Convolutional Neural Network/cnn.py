import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# Build Fully Connected Neural Network (DNN)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
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

# Plot Accuracy Curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label="Training Accuracy", marker='o')
plt.plot(history.history['val_accuracy'], label="Validation Accuracy", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.show()

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Training Loss", marker='o', color='r')
plt.plot(history.history['val_loss'], label="Validation Loss", marker='s', color='g')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()
