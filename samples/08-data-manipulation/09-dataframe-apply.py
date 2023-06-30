import pandas as pd

# Create a DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Apply a custom function to each column
col_sum = df.apply(lambda x: x.sum(), axis=0)

print(col_sum)
