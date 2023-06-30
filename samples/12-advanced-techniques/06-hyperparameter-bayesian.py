import skopt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

svm = SVC()

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter search space
search_space = {
    'C': Real(1e-3, 1e3, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf', 'sigmoid']),
    'gamma': Real(1e-3, 1e3, prior='log-uniform')
}

# Perform Bayesian optimization
bayes_search = BayesSearchCV(estimator=svm, search_spaces=search_space, n_iter=10, cv=5)
bayes_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", bayes_search.best_params_)
