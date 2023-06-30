import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data     # Independent variables
y = iris.target   # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('X_train')
print(X_train[0:5])
print('X_test')
print(X_test[0:5])
print('y_train')
print(y_train)
print('y_test')
print(y_test)
