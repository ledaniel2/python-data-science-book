import seaborn as sns
import matplotlib.pyplot as plt

# Load the built-in 'penguins' dataset
penguins = sns.load_dataset('penguins')

# Create a scatter plot
sns.scatterplot(data=penguins, x='body_mass_g', y='bill_length_mm', hue='species')

# Add a title
plt.title('Penguins dataset scatter plot')

# Display the plot
plt.show()
