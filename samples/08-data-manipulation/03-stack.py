import pandas as pd

# Creating a wide-format DataFrame
data = {
    '2021': [10, 20],
    '2022': [11, 21]
}
wide_df = pd.DataFrame(data, index=['A', 'B'])

# Stacking the DataFrame
stacked_df = wide_df.stack()

print(stacked_df)
