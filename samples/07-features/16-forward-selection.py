from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# Instantiate the model and the forward selector
model = LinearRegression()
selector = SequentialFeatureSelector(model, k_features=4, forward=True)

# Fit and transform the dataset
X_selected_features = selector.fit_transform(X, y)

print('X_selected_features:')
print(X_selected_features[0:5])
