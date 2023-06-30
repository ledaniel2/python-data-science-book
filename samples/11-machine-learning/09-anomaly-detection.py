from sklearn.ensemble import IsolationForest
import numpy as np

# Create random data
X = np.random.rand(100, 2)

# Initialize IsolationForest object
clf = IsolationForest()

# Fit the data to the IsolationForest object
clf.fit(X)

# Get the anomaly scores for each data point
scores = clf.decision_function(X)

print(scores)
