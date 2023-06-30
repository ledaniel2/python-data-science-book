import pandas as pd
import numpy as np

data = {
    'A': [1, 2, 3, 4, 100],
    'B': [5, 6, 7, 8, 200],
    'C': [9, 10, 11, 12, 300],
}

df = pd.DataFrame(data)

# Calculate Z-scores
z_scores = np.abs((df - df.mean()) / df.std())

# Identify outliers using the Z-score
outliers = z_scores > 3
print(outliers)

# Calculate the IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Identify outliers using the IQR
outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
print(outliers)

# Remove outliers using the Z-score method
outlier_indices = np.where(z_scores > 3)
df_no_outliers = df.drop(df.index[outlier_indices[0]])
print(df_no_outliers)

# Remove outliers using the IQR method
df_no_outliers2 = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df_no_outliers2)

# Apply a logarithmic transformation
df_log = np.log(df)
print(df_log)

# Cap outliers using the mean
mean_values = df.mean()
df_capped = df.where(~outliers, mean_values, axis=1)
print(df_capped)

# Cap outliers using the median
median_values = df.median()
df_capped2 = df.where(~outliers, median_values, axis=1)
print(df_capped2)

# Cap outliers with the maximum and minimum non-outlier values
min_values = Q1 - 1.5 * IQR
max_values = Q3 + 1.5 * IQR

df_capped3 = df.copy()
df_capped3[outliers] = np.where(df < min_values, min_values, max_values)
print(df_capped3)
