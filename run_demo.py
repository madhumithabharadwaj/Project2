import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model.BoostingTree import BoostingTree


#  Pure NumPy Data Generator
def generate_binary_classification_data(n_samples=350, n_features=5, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)

    # Create "true" weights and generate labels
    true_weights = np.random.randn(n_features)
    logits = X @ true_weights
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)  # Convert to binary labels

    return X, y

# Generate dataset
X, y = generate_binary_classification_data()

#
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["label"] = y
df.to_csv("generated_data.csv", index=False)

# Train your BoostingTree model
model = BoostingTree(n_estimators=100, learning_rate=0.1, early_stopping_rounds=5)
model.fit(X, y)

# Predict and evaluate
predictions = model.predict(X)
accuracy = (predictions == y).mean()

# Display results
print(" Accuracy on training data:", accuracy)
print("Predictions (first 10):", predictions[:10])
print("True labels     (first 10):", y[:10])

##plotting the saved data:

# Visualize 2D projection: feature_0 vs feature_1
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("Generated Binary Classification Data")
plt.grid(True)
plt.show()



# Bonus 1: Accuracy vs. n_estimators Curve
accuracies = []

# Vary number of estimators and record training accuracy
for n in range(1, 51, 5):  # Try 1, 6, 11, ..., 46
    model = BoostingTree(n_estimators=n, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    acc = (preds == y).mean()
    accuracies.append((n, acc))

# Plot accuracy curve
plt.figure(figsize=(8, 5))
plt.plot([a[0] for a in accuracies], [a[1] for a in accuracies], marker='o')
plt.xlabel("Number of Estimators")
plt.ylabel("Training Accuracy")
plt.title("Boosting Performance vs. Number of Estimators")
plt.grid(True)
plt.show()



# Bonus 2: Visualize Model Predictions
# Plot predicted labels instead of true labels
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("Model Predictions (color = predicted class)")
plt.grid(True)
plt.show()