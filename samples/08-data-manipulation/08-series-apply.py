import pandas as pd

# Create a Series
s = pd.Series([1, 2, 3, 4])

# Apply a custom function to each element
s_squared = s.apply(lambda x: x**2)

print(s_squared)
