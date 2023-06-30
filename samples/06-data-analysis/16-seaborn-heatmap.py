import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
flights = sns.load_dataset('flights')
flights = flights.pivot(index='month', columns='year', values='passengers')
ax = sns.heatmap(flights)
plt.title('Heatmap Flight Data')
plt.show()
