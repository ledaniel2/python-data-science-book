import seaborn as sns
import matplotlib.pyplot as plt

# Load the built-in 'iris' dataset
iris = sns.load_dataset('iris')

# Create a scatter plot
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species')

# Add a title
plt.title('Iris dataset scatter plot')

# Display the plot
plt.show()
