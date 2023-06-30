import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# Filter DataFrame by age
filtered_df = df[df['Age'] > 25]
print(filtered_df)
