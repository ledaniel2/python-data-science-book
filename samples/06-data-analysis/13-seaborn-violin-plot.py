import seaborn as sns
import matplotlib.pyplot as plt

# Load the built-in 'iris' dataset
iris = sns.load_dataset('iris')

# Create a violin plot
sns.violinplot(data=iris, x='species', y='sepal_length')

# Add a title
plt.title('Iris dataset violin plot')

# Display the plot
plt.show()
