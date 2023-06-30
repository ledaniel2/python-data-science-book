import pandas as pd

data = pd.read_excel('data.xlsx', sheet_name='Sheet1')
print(data.head())
