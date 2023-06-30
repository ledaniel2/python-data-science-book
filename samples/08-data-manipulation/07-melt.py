import pandas as pd

# Create a wide-format DataFrame
data = {
    'ID': [1, 2, 3],
    'Category_A': [10, 11, 12],
    'Category_B': [20, 21, 22],
    'Category_C': [30, 31, 32]
}
wide_df = pd.DataFrame(data)

# Melt the DataFrame
long_df = pd.melt(wide_df, id_vars=['ID'], value_vars=['Category_A', 'Category_B', 'Category_C'],
                  var_name='Category', value_name='Value')

print(long_df)
