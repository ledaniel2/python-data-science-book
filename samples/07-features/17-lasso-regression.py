from sklearn.linear_model import Lasso
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# Instantiate the Lasso model with an alpha value
model = Lasso(alpha=0.1)

# Fit the model
model.fit(X, y)

# Select features with non-zero coefficients
X_selected_features = X[:, model.coef_ != 0]

print('X_selected_features:')
print(X_selected_features[0:5])
