import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    'A': [1, 2, 3, 4],
    'B': [100, 200, 300, 400],
    'C': [1000, 2000, 3000, 4000],
}

df = pd.DataFrame(data)

scaler = StandardScaler()
standardized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(standardized_df)