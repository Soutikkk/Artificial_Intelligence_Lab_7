# KNN - Iris Flower Classification
# Basic Beginner Program

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("KNN - Iris Flower Classification")
print("--------------------------------")
print("Accuracy:", accuracy)

# Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = knn.predict(new_flower)

print("Predicted Flower:", iris.target_names[prediction[0]])
