import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv'
data = pd.read_csv(url)

# Check the distribution of loan approvals by gender
sns.countplot(x='Loan_Status', hue='Gender', data=data)

# Show the bar plot
plt.show()

from sklearn.utils import resample

# Separate majority and minority classes
data_majority = data[data['Loan_Status'] == 'Y']
data_minority = data[data['Loan_Status'] == 'N']

# Upsample minority class
data_minority_upsampled = resample(data_minority, replace=True, n_samples=len(data_majority), random_state=42)

# Combine majority class with upsampled minority class
balanced_data = pd.concat([data_majority, data_minority_upsampled])
print(balanced_data)

# One-hot encode categorical variables
encoded_data = pd.get_dummies(data, columns=['Gender', 'Property_Area'])

# Encode the data as numbers
from sklearn.preprocessing import LabelEncoder
X = encoded_data.drop('Loan_Status', axis=1)
le = LabelEncoder()
for x in X.columns:
    X[x] = le.fit_transform(X[x])

y = le.fit_transform(encoded_data['Loan_Status'])

# Perform feature selection using ANOVA F-value
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Get the selected features
selected_features = X.columns[selector.get_support()]
print(selected_features)
