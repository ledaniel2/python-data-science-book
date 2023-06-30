import pandas as pd

# Create a sample dataset
data = {'A': [1, 2, 3, 3, 5], 'B': [10, 20, 30, 30, 50]}
df = pd.DataFrame(data)

# Calculate the mean
mean_A = df['A'].mean()
mean_B = df['B'].mean()

print('Mean of A:', mean_A)
print('Mean of B:', mean_B)

# Calculate the median
median_A = df['A'].median()
median_B = df['B'].median()

print('Median of A:', median_A)
print('Median of B:', median_B)
