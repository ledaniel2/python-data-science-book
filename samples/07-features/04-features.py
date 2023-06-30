import pandas as pd

# Example DataFrame
data = pd.DataFrame({'height': [160, 170, 180], 'weight': [65, 75, 85]})

data['bmi'] = data['weight'] / (data['height'] / 100) ** 2

print(data)
