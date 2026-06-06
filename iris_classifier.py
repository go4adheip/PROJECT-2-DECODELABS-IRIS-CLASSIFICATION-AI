import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load Iris Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

print("=== First 5 Rows of Dataset ===")
print(df.head())

# Features and Target
X = iris.data
y = iris.target

# Split Dataset (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN Model
model = KNeighborsClassifier(n_neighbors=5)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Accuracy ===")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n=== Confusion Matrix ===")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix - Iris Classification")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.savefig("confusion_matrix.png")
print("\nConfusion matrix plot saved as 'confusion_matrix.png'.")

# Show the plot briefly without blocking forever
try:
    plt.show(block=False)
    plt.pause(2)
    plt.close()
except Exception:
    pass

# Example Prediction
sample_flower = [[5.1, 3.5, 1.4, 0.2]]

sample_flower = scaler.transform(sample_flower)

prediction = model.predict(sample_flower)

print("\n=== New Flower Prediction ===")
print("Predicted Species:", iris.target_names[prediction[0]])