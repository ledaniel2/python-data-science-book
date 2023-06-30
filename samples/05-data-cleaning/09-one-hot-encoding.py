import pandas as pd

data = {
    'Color': ['Red', 'Green', 'Blue', 'Red', 'Green'],
    'Size': ['S', 'M', 'L', 'XL', 'XXL'],
}

df = pd.DataFrame(data)

# One-hot encoding using pandas
one_hot_df = pd.get_dummies(df, columns=['Color'])
print(one_hot_df)
