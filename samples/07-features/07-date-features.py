import pandas as pd

# Example date data
date_data = pd.DataFrame({'date': pd.date_range(start='2020-01-01', periods=5, freq='D')})

# Extracting date features
date_data['day_of_week'] = date_data['date'].dt.dayofweek
date_data['month'] = date_data['date'].dt.month
date_data['year'] = date_data['date'].dt.year

print(date_data)
