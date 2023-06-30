from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Instantiate a Logistic Regression model
model = LogisticRegression()

# Create the RFE object with the desired number of features
rfe = RFE(model, n_features_to_select=2)

# Fit the RFE object on the dataset
rfe.fit(X_train, y_train)

# Print the selected features
print("Selected features:")
for i in range(X.shape[1]):
    if rfe.support_[i]:
        print(f"Feature {i + 1}")
