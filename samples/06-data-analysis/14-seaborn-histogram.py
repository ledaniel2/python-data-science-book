import seaborn as sns
import matplotlib.pyplot as plt

# Load the built-in 'iris' dataset
iris = sns.load_dataset('iris')

# Create a histogram
sns.histplot(data=iris, x='sepal_length', kde=True)

# Add a title
plt.title('Iris dataset histogram')

# Display the plot
plt.show()
