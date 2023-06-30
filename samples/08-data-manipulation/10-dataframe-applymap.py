import pandas as pd

# Create a DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Apply a custom function to each element
df_squared = df.applymap(lambda x: x**2)

print(df_squared)
