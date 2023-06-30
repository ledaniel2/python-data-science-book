import numpy as np
import matplotlib.pyplot as plt

# Create sample data
data = np.random.randn(1000)

# Create a histogram
plt.hist(data, bins=30)

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('A histogram')

# Display the plot
plt.show()
