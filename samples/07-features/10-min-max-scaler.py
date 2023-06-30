import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Example dataset
data = np.array([[100, 0.5],
                 [80, 0.4],
                 [120, 0.6],
                 [90, 0.55]])

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)
