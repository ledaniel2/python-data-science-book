import pandas as pd

# Create a sample dataset
data = {'A': [1, 2, 2, 3, 5], 'B': [10, 30, 30, 30, 50]}
df = pd.DataFrame(data)

# Calculate the skewness
skewness_A = df['A'].skew()
skewness_B = df['B'].skew()

print('Skewness of A:', skewness_A)
print('Skewness of B:', skewness_B)

# Calculate the kurtosis
kurtosis_A = df['A'].kurt()
kurtosis_B = df['B'].kurt()

print('Kurtosis of A:', kurtosis_A)
print('Kurtosis of B:', kurtosis_B)
