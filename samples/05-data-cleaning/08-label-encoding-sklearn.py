import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    'Color': ['Red', 'Green', 'Blue', 'Red', 'Green'],
    'Size': ['S', 'M', 'L', 'XL', 'XXL'],
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['Color_encoded'] = le.fit_transform(df['Color'])
df['Size_encoded'] = le.fit_transform(df['Size'])

print(df)