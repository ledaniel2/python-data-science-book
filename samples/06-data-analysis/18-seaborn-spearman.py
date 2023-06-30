import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the sample dataset
data = sns.load_dataset('iris')

# Calculate Spearman's rank correlation coefficient
spearman_matrix = data.corr(method='spearman', numeric_only=True)

# Visualize the Spearman's rank matrix using a heatmap
sns.heatmap(spearman_matrix, annot=True, cmap='coolwarm')
plt.show()
