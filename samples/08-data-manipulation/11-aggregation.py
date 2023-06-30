import pandas as pd

# Create a DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Apply multiple aggregation functions to each column
result = df.agg(['sum', 'mean'], axis=0)

print(result)
