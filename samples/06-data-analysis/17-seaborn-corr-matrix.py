import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the sample dataset
data = sns.load_dataset('iris')

# Calculate the correlation matrix
corr_matrix = data.corr()

# Visualize the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
