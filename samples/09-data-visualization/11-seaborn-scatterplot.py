import seaborn as sns
import matplotlib.pyplot as plt

# Load the example dataset
data = sns.load_dataset('tips')
# Create a scatterplot with customized point style, color, and size
sns.scatterplot(x='total_bill', y='tip', data=data, marker='D', s=100, color='purple', label='Tips')

# Customize plot appearance
plt.title('Customized Tips Scatterplot')
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.legend(loc='upper right')

# Show the plot
plt.show()
