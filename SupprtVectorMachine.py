import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading dataset
dataset = pd.read_csv(" <filename> ")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Reshape y to a 2D array for the StandardScaler
y = y.reshape(len(y), 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
# Fix: You need two separate instances of StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Training the SVR model
from sklearn.svm import SVR
# Fix: kernel must be lowercase 'kernel'
regressor = SVR(kernel='rbf')
# Use .ravel() to convert y back to a 1D array for the fit method
regressor.fit(X, y.ravel())

# Predicting a new result
# Fix: Predict using sc_X scaled input, then inverse_transform the result
y_pred_scaled = regressor.predict(sc_X.transform([[6.5]]))
# Reshape to 2D for inverse_transform
y_pred_final = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

print(f"Predicted Value: {y_pred_final}")
