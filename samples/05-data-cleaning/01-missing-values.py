import pandas as pd

data = {
    'A': [1, 2, None, 4],
    'B': [5, None, 7, 8],
    'C': [9, 10, 11, None],
}

df = pd.DataFrame(data)
print(df.isna())

# Remove rows containing missing values
df_no_missing = df.dropna()
print(df_no_missing)

# Remove columns containing missing values
df_no_missing_columns = df.dropna(axis=1)
print(df_no_missing_columns)

mean_imputed_df = df.fillna(df.mean())
print(mean_imputed_df)

median_imputed_df = df.fillna(df.median())
print(median_imputed_df)

mode_imputed_df = df.fillna(df.mode().iloc[0])
print(mode_imputed_df)

interpolated_df = df.interpolate()
print(interpolated_df)

forward_df = df.fillna(method='ffill')
print(forward_df)
backward_df = df.fillna(method='bfill')
print(backward_df)

# Fill missing values with a custom value
custom_imputed_df = df.fillna(-1)
print(custom_imputed_df)

# Custom imputation function
def custom_impute(column):
    return column.fillna(column.mean())

# Apply custom imputation function to each column
custom_imputed_df2 = df.apply(custom_impute)
print(custom_imputed_df2)
