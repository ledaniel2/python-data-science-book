import numpy as np
import matplotlib.pyplot as plt

# Create sample data
x = ['Category A', 'Category B', 'Category C']
y = [25, 45, 30]

# Create a bar plot
plt.bar(x, y)

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('A bar plot')

# Display the plot
plt.show()
