import seaborn as sns
import matplotlib.pyplot as plt

# Load the built-in 'iris' dataset
iris = sns.load_dataset('iris')

# Create a pair plot
sns.pairplot(data=iris, hue='species')

# Add a title
plt.suptitle('Iris dataset pair plot', y=1.02)

# Display the plot
plt.show()
