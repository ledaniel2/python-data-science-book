import pandas as pd

# Create a DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Year': [2021, 2022, 2021, 2022, 2021, 2022],
    'Value': [10, 11, 20, 21, 12, 22]
}
df = pd.DataFrame(data)

# Group the data by 'Category'
grouped = df.groupby('Category')

# Calculate the mean value for each group
mean_values = grouped['Value'].mean()

print(mean_values)
