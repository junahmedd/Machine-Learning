import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset = pd.read_csv("<filename>")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values  # Standard convention uses lowercase y for vectors

# 1. Handle Missing Data
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# 2. Encoding Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# 3. Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 4. Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Note: Scaling usually applies to all numerical features. 
# Ensure index 3: is correct for your specific dataset after OneHotEncoding.
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("Preprocessed X_train sample:\n", X_train[:5])
