import pandas as pd

# Data
data = [
    ('A', 'X', 1),
    ('A', 'Y', 2),
    ('A', 'Z', 3),
    ('B', 'X', 4),
    ('B', 'Y', 5),
    ('B', 'Z', 6)
]

# Create a MultiIndex DataFrame
index = pd.MultiIndex.from_tuples([(row[0], row[1]) for row in data], names=['Letter', 'Coordinate'])
df = pd.DataFrame([row[2] for row in data], index=index, columns=['Value'])

print(df)

# Access data for letter 'A'
print(df.loc['A'])

# Access data for letter 'A' and coordinate 'Y'
print(df.loc['A','Y'])
