import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Calculate the mean
mean = np.mean(data)
print("Mean:", mean)

# Calculate the median
median = np.median(data)
print("Median:", median)

# Calculate the standard deviation
std_dev = np.std(data)
print("Standard deviation:", std_dev)

# Calculate the correlation coefficient
data_1 = np.array([1, 2, 3, 4, 5])
data_2 = np.array([2, 4, 6, 8, 10])
corr_coef = np.corrcoef(data_1, data_2)
print("Correlation coefficient:\n", corr_coef)
