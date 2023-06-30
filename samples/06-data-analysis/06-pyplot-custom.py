import numpy as np
import matplotlib.pyplot as plt

# Create sample data
x = np.arange(0, 10, 0.1)
y = np.sin(x)

# Create a line plot with custom color, marker, and line style
plt.plot(x, y, color='red', marker='o', linestyle='--', linewidth=2)

# Add labels and title
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('A customized sine wave plot')

# Display the plot
plt.show()
