import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

sns.lineplot(x=x, y=y, linestyle='dashdot')
plt.show()
