import plotly.express as px
import pandas as pd

# Create a sample dataset
data = pd.DataFrame({'X': range(1, 11), 'Y': [i**2 for i in range(1, 11)]})
# Create an interactive scatter plot
fig = px.scatter(data, x='X', y='Y', title='Interactive Scatter Plot', labels={'X': 'X-axis', 'Y': 'Y-axis'})

# Show the plot in browser
fig.show()
