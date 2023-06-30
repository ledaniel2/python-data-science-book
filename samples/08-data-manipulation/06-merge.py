import pandas as pd

# Create two DataFrames
data1 = {'Key': ['A', 'B', 'C'], 'Value1': [1, 2, 3]}
data2 = {'Key': ['B', 'C', 'D'], 'Value2': [4, 5, 6]}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Merge the DataFrames on the 'Key' column
merged_df = pd.merge(df1, df2, on='Key', how='inner')

print(merged_df)
