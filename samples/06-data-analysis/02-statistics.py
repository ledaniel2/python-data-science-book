import pandas as pd

# Create a sample dataset with multiple modes
data = {'A': [1, 2, 3, 3, 4], 'B': [10, 20, 30, 30, 40]}
df = pd.DataFrame(data)

# Calculate the mode
mode_A = df['A'].mode()
mode_B = df['B'].mode()

print('Mode of A:', mode_A)
print('Mode of B:', mode_B)

# Calculate the range
range_A = df['A'].max() - df['A'].min()
range_B = df['B'].max() - df['B'].min()

print('Range of A:', range_A)
print('Range of B:', range_B)

# Calculate the variance
variance_A = df['A'].var()
variance_B = df['B'].var()

print('Variance of A:', variance_A)
print('Variance of B:', variance_B)

# Calculate the standard deviation
std_dev_A = df['A'].std()
std_dev_B = df['B'].std()

print('Standard Deviation of A:', std_dev_A)
print('Standard Deviation of B:', std_dev_B)

# Calculate the IQR
Q1_A = df['A'].quantile(0.25)
Q3_A = df['A'].quantile(0.75)
IQR_A = Q3_A - Q1_A

Q1_B = df['B'].quantile(0.25)
Q3_B = df['B'].quantile(0.75)
IQR_B = Q3_B - Q1_B

print('Interquartile Range of A:', IQR_A)
print('Interquartile Range of B:', IQR_B)
