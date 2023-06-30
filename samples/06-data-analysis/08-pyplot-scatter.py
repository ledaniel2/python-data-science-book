import numpy as np
import matplotlib.pyplot as plt

# Create sample data
x = np.random.rand(50)
y = np.random.rand(50)

# Create a scatter plot
plt.scatter(x, y)

# Add labels and title
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('A scatter plot')

# Display the plot
plt.show()
