from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.svm import SVC

svm = SVC()

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter search space
param_dist = {
    'C': uniform(loc=0, scale=4),
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': uniform(loc=0, scale=4)
}

# Perform random search
random_search = RandomizedSearchCV(estimator=svm, param_distributions=param_dist, n_iter=100, cv=5)
random_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", random_search.best_params_)
