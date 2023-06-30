import pandas as pd

data = {
    'A': [1, 2, 3, 4],
    'B': [100, 200, 300, 400],
    'C': [1000, 2000, 3000, 4000],
}

df = pd.DataFrame(data)

# Standardize the DataFrame
standardized_df = (df - df.mean()) / df.std()
print(standardized_df)
