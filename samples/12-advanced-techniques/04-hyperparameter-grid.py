import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load the dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter search space
param_grid = {
    'C': np.logspace(-3, 3, 7),
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': np.logspace(-3, 3, 7)
}

# Initialize the model
svm = SVC()

# Perform grid search
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
