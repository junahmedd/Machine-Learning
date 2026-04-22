import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Loading dataset
dataset = pd.read_csv(" <filename> ")
X = dataset.iloc[:, :-1].values # Changed to capital X (standard for features)
y = dataset.iloc[:, -1].values

# Splitting the data
from sklearn.model_selection import train_test_split
# Fix: Used capital X to match the definition above
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0, test_size=0.3)

# Training the Linear Regression model
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regression.predict(X_test)

# Visualisation (Training Set)
plt.scatter(X_train, Y_train, color="red")
# The blue line represents the model's prediction (the best fit line)
plt.plot(X_train, regression.predict(X_train), color="blue")

# Fix: xlabel and ylabel must be lowercase
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience (Training set)")
plt.show()
