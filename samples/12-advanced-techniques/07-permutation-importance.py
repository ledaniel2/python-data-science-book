from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn import datasets

# Load the dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Load dataset and split it into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Calculate feature importances using permutation importance
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)

# Print feature importances
for i in range(X.shape[1]):
    print(f"Feature {i + 1}: {result.importances_mean[i]:.3f} +/- {result.importances_std[i]:.3f}")
