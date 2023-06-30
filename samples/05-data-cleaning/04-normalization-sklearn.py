import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    'A': [1, 2, 3, 4],
    'B': [100, 200, 300, 400],
    'C': [1000, 2000, 3000, 4000],
}

df = pd.DataFrame(data)

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)

normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
print(normalized_df)
