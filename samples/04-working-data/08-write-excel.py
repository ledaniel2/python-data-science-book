import pandas as pd

data = pd.read_excel('data.xlsx', sheet_name='Sheet1')

data.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
print('Written file: output.xlsx')
