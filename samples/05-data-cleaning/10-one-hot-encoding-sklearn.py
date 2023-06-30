import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {
    'Color': ['Red', 'Green', 'Blue', 'Red', 'Green'],
    'Size': ['S', 'M', 'L', 'XL', 'XXL'],
}

df = pd.DataFrame(data)

ohe = OneHotEncoder()
encoded_array = ohe.fit_transform(df[['Color']]).toarray()
encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(['Color']))
one_hot_df = pd.concat([df.drop('Color', axis=1), encoded_df], axis=1)
print(one_hot_df)
