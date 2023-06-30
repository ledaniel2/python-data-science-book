import numpy as np
import matplotlib.pyplot as plt

# Create sample data
x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(x)

# Create the first subplot
plt.subplot(2, 2, 1)
plt.plot(x, y1)
plt.title('Sine wave')

# Create the second subplot
plt.subplot(2, 2, 2)
plt.plot(x, y2)
plt.title('Cosine wave')

# Create the third subplot
plt.subplot(2, 2, 3)
plt.plot(x, y3)
plt.title('Tangent wave')

# Create the fourth subplot
plt.subplot(2, 2, 4)
plt.plot(x, y4)
plt.title('Exponential wave')

# Adjust the layout
plt.tight_layout()

# Display the plots
plt.show()
