import pandas as pd

data = {
    'Color': ['Red', 'Green', 'Blue', 'Red', 'Green'],
    'Size': ['S', 'M', 'L', 'XL', 'XXL'],
}

df = pd.DataFrame(data)

# Label encoding using pandas
df['Color_encoded'] = df['Color'].astype('category').cat.codes
df['Size_encoded'] = df['Size'].astype('category').cat.codes
print(df)
