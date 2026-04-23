import pandas as pd 
import numpy as np

# Loading dataset
dataset = pd.read_csv(" <filename> ")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X) # Standardize back to 'X'

# Splitting the data
from sklearn.model_selection import train_test_split
# Fix: Used capital X and lowercase y to match defined variables
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Training the Artificial Neural Network
from sklearn.neural_network import MLPClassifier # Fix: neural_network (with underscore)
# Fix: hidden_layer_sizes (plural + underscores)
# Fix: batch_size and max_iter should be integers, not strings
model = MLPClassifier(hidden_layer_sizes=(6, 6),
                      activation='relu',
                      solver='adam',
                      batch_size=32,
                      max_iter=200,
                      random_state=0)

model.fit(X_train, Y_train)

# Evaluation
accuracy = model.score(X_test, Y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
