import pandas as pd

# Creating a long-format DataFrame with a MultiIndex
index = pd.MultiIndex.from_tuples([('A', 2021), ('A', 2022), ('B', 2021), ('B', 2022)])
data = [10, 11, 20, 21]
long_df = pd.Series(data, index=index)

# Unstacking the DataFrame
unstacked_df = long_df.unstack()

print(unstacked_df)
