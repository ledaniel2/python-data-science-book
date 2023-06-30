import pandas as pd

data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 45],
    'city': ['New York', 'San Francisco', 'Los Angeles', 'Seattle', 'Chicago']
}

df = pd.DataFrame(data)
print(df)
# Selecting a single column
print(df['name'])

# Selecting multiple columns
print(df[['name', 'city']])

# Selecting a row by index
print(df.loc[2])

# Selecting a row by row number
print(df.iloc[2])

# Slicing rows using only rows 1-3 inclusive
print(df[1:4])
# Filter rows where age is greater than 30
print(df[df['age'] > 30])
# Filter rows where age is greater than 30 and city is 'Los Angeles'
print(df[(df['age'] > 30) & (df['city'] == 'Los Angeles')])
# Sort by age, ascending
print(df.sort_values('age'))

# Sort by age, descending
print(df.sort_values('age', ascending=False))

# Sort by multiple columns
print(df.sort_values(['city', 'age']))
# Add a new column
df['name_and_city'] = df['name'] + ' from ' + df['city']
print(df)
# Drop a single column
print(df.drop('age', axis=1))

# Drop multiple columns
print(df.drop(['age', 'name_and_city'], axis=1))
# Drop a single row by index
print(df.drop(0, axis=0))

# Drop multiple rows by index
print(df.drop([0, 1, 2], axis=0))
# Drop rows where a column meets a condition
print(df[df['city'] != 'New York'])
