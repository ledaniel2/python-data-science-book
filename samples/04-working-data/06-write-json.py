import pandas as pd

data = pd.read_json('data.json')

data.to_json('output.json', orient='records', lines=True)
print('Written file: output.json')
