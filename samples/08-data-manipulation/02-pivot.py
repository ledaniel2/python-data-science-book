import pandas as pd

# Creating a long-format DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B'],
    'Year': [2021, 2022, 2021, 2022],
    'Value': [10, 11, 20, 21]
}
long_df = pd.DataFrame(data)

# Pivoting the DataFrame
wide_df = long_df.pivot(index='Category', columns='Year', values='Value')

print(wide_df)
