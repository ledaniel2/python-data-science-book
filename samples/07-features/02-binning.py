import numpy as np
import pandas as pd

# Example DataFrame
data = pd.DataFrame({'age': [2, 9, 12, 17, 21, 22, 24, 30, 35, 41, 49, 50, 55, 70, 89]})

# Create age categories using custom bins
bins = [0, 18, 35, 60, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Senior']
data['age_category'] = pd.cut(data['age'], bins=bins, labels=labels)
print(data)

# Equal-width binning
data['value_bin_eq_width'] = pd.cut(data['age'], bins=3)

# Equal-frequency binning
data['value_bin_eq_freq'] = pd.qcut(data['age'], q=3)

print(data)
