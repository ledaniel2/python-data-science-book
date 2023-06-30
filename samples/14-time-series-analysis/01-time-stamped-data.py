import pandas as pd
import numpy as np

# Generate a range of dates
date_rng = pd.date_range(start='1/1/2022', end='12/31/2022', freq='D')

# Create a time series DataFrame with random temperature values
temperature_data = np.random.randint(50, 100, size=(len(date_rng)))
temp_df = pd.DataFrame(date_rng, columns=['date'])
temp_df['temperature'] = temperature_data

print(temp_df.head())

temp_df.set_index('date', inplace=True)
print(temp_df.head())

monthly_avg_temp = temp_df.resample('M').mean()
print(monthly_avg_temp)

feb_temp = temp_df['2022-02-01':'2022-02-28']
print(feb_temp)

temp_df['previous_day_temp'] = temp_df['temperature'].shift(1)
temp_df['daily_change'] = temp_df['temperature'] - temp_df['previous_day_temp']
print(temp_df.head())

# Introduce a gap in the data
temp_df.iloc[7:10, 0] = np.nan

# Fill the gap using forward fill
temp_df['temperature_filled'] = temp_df['temperature'].fillna(method='ffill')
print(temp_df.head(12))
