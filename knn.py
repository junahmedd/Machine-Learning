# Load and split
from sklearn.datasets import load_iris
x, y = load_iris(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Training KNN model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5).fit(X_train, Y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy: ", accuracy_score(Y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("\nClassification Report:\n", classification_report(Y_test, y_pred))

# Visualisation 
import matplotlib.pyplot as plt
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, edgecolor='k', alpha=0.7)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("KNN on Iris (Test Set)")
plt.show()
