import pandas as pd

data = {
    'Color': ['Red', 'Green', 'Blue', 'Red', 'Green'],
    'Size': ['S', 'M', 'L', 'XL', 'XXL'],
}

df = pd.DataFrame(data)

# Dummy encoding using pandas
dummy_df = pd.get_dummies(df, columns=["Color"], drop_first=True)
print(dummy_df)
