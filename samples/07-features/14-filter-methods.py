from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# Instantiate the selector with a threshold
selector = VarianceThreshold(threshold=0.2)

# Fit and transform the dataset
X_high_variance = selector.fit_transform(X)

print('X_high_variance:')
print(X_high_variance[0:5])

# Instantiate the selector, selecting the top 2 features
selector = SelectKBest(chi2, k=2)

# Fit and transform the dataset
X_best_features = selector.fit_transform(X, y)

print('X_best_features:')
print(X_best_features[0:5])
