import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample data
data = np.array([[10, 200], [15, 180], [30, 210]])

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(data)
print(scaled_data)
