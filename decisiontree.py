import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading dataset
dataset = pd.read_csv(" <filename> ")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the data
# Fix: model_selection (added missing dot)
from sklearn.model_selection import train_test_split
# Fix: Ensure variable case matches (X instead of x)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training ID3 (Decision Tree using Entropy)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train, Y_train)

# Predict a single observation
# Fix: use .predict() method (added dot, removed underscore)
print("Single Prediction:", classifier.predict(sc.transform([[30, 87500]])))

# Predict test set
y_pred = classifier.predict(X_test)

# Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score # Fix: metrics (added 's')
cm = confusion_matrix(Y_test, y_pred) # Fix: variable names case (Y_test)
print("Confusion Matrix:\n", cm)
print("Accuracy Score: ", accuracy_score(Y_test, y_pred))
