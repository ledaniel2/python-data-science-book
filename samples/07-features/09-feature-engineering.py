import pandas as pd

# Example DataFrame
data = pd.DataFrame({'year_built': [1990, 2000, 2010, 2020]})

# Using domain knowledge to create a new feature
current_year = pd.Timestamp.now().year
data['age'] = current_year - data['year_built']

print(data)