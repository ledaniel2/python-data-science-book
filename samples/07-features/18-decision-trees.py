from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# Instantiate the Random Forest model
model = RandomForestClassifier()

# Fit the model
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Set a threshold for feature importance
threshold = 0.1

# Select features with importance greater than the threshold
X_selected_features = X[:, importances > threshold]

print('X_selected_features:')
print(X_selected_features[0:5])
