import numpy as np
import matplotlib.pyplot as plt

# Create a sample dataset
x = np.linspace(0, 10, 50)
y = np.sin(x)

# Create a plot with customized line style, color, and marker
plt.plot(x, y, linestyle='dashed', color='lightgreen', marker='o', label='Sine Wave')

# Customize plot appearance
plt.title('Customized Sine Wave Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='lower right')

# Use a grid
plt.grid(color='black')

# Show the plot
plt.show()
