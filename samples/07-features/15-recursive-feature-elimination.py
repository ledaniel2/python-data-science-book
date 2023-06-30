from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# Instantiate the model and the RFE selector
model = LogisticRegression(max_iter=1000)
selector = RFE(model, n_features_to_select=3)

# Fit and transform the dataset
X_selected_features = selector.fit_transform(X, y)

print('X_selected_features:')
print(X_selected_features[0:5])
