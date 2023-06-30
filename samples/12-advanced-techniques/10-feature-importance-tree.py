from sklearn.ensemble import RandomForestClassifier

# Train a RandomForest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Get feature importances
importances = clf.feature_importances_

# Print feature importances
for i in range(X.shape[1]):
    print(f"Feature {i + 1}: {importances[i]:.3f}")
import matplotlib.pyplot as plt

# Sort importances and their corresponding feature indices
sorted_idx = importances.argsort()

# Plot the feature importances
plt.barh(range(X.shape[1]), importances[sorted_idx])
plt.yticks(range(X.shape[1]), [f"Feature {i + 1}" for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.show()
