import pandas as pd

data = pd.read_csv('data.csv')

data.to_csv('output.csv', index=False)
print('Written file: output.csv')
