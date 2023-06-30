import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

#plt.figure(figsize=(12, 6))
#plt.plot(data)
#plt.xlabel('Month')
#plt.ylabel('Number of Passengers')
#plt.title('Airline Passengers (1949-1960)')
#plt.show()

decomposition = seasonal_decompose(data, model='multiplicative')

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
ax1.plot(data)
ax1.set_title('Original Time Series')
ax1.set_xlabel('Month')
ax1.set_ylabel('Number of Passengers')

ax2.plot(decomposition.trend)
ax2.set_title('Trend')
ax2.set_xlabel('Month')
ax2.set_ylabel('Number of Passengers')

ax3.plot(decomposition.seasonal)
ax3.set_title('Seasonal')
ax3.set_xlabel('Month')
ax3.set_ylabel('Number of Passengers')

ax4.plot(decomposition.resid)
ax4.set_title('Residual')
ax4.set_xlabel('Month')
ax4.set_ylabel('Number of Passengers')

plt.tight_layout()
plt.show()
