import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Loading dataset
dataset = pd.read_csv(" <filename> ")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
features = dataset.columns[:-1]

# Splitting the data
from sklearn.model_selection import train_test_split
# Fix: Ensure y matches the case defined above
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Fix: fit_transform (added 'n')
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10,
                               criterion='entropy',
                               random_state=0)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

# Evaluation
# Fix: metrics (added 's')
from sklearn.metrics import accuracy_score, confusion_matrix
# Fix: Ensure Y_test matches the case defined in split
cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy Score: ", accuracy_score(Y_test, y_pred))

# Visualisation of Feature Importance
# Fix: plt.barh (horizontal bar chart); feature_importances_ (underscore at the end)
plt.barh(features, model.feature_importances_)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest")
plt.show()
