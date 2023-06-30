import seaborn as sns
import matplotlib.pyplot as plt

# Load the 'iris' dataset
iris = sns.load_dataset('iris')

# Create a box plot
sns.boxplot(data=iris, x='species', y='sepal_length')

# Add a title
plt.title('Iris dataset box plot')

# Display the plot
plt.show()
