import pandas as pd

# Example DataFrame
data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Multiplication
data['AB_mult'] = data['A'] * data['B']

# Division
data['AB_div'] = data['A'] / data['B']

# Addition
data['AB_add'] = data['A'] + data['B']

# Subtraction
data['AB_sub'] = data['A'] - data['B']

print(data)
