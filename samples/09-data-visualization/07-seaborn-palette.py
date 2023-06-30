import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': [2, 4, 5, 5, 4, 6], 'z': [0, 0, 1, 2, 1, 2]})
sns.lineplot(df, x='x', y='y', hue='z', palette='coolwarm')
plt.show()
