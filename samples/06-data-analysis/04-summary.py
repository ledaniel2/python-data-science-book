import pandas as pd

# Create a sample dataset
data = {'A': [1, 2, 2, 3, 5], 'B': [10, 30, 30, 30, 50]}
df = pd.DataFrame(data)

# Generate summary statistics for the DataFrame
summary_stats = df.describe()
print(summary_stats)
