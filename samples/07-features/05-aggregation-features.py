import pandas as pd

# Example DataFrame
data = pd.DataFrame({'category': ['A', 'A', 'B', 'B', 'A', 'B'],
                     'value': [1, 2, 3, 4, 5, 6]})

# Aggregation features
agg_features = data.groupby('category')['value'].agg(['sum', 'mean', 'min', 'max', 'count']).reset_index()
print(agg_features)
