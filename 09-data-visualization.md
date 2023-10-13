# Chapter 9: Advanced Data Visualization

Visualizing data is a crucial aspect of data science, as it allows us to communicate complex information in an accessible and engaging manner. Effective data visualizations can not only help you explore and understand your data but also enable you to share your findings with others in a clear and impactful way.

We will build upon the foundations laid in chapter 6, diving deeper into the customization of plots using the `matplotlib` and `seaborn` libraries. You will learn how to modify plot styles, colors, labels, and other elements, allowing you to create visualizations that align with your specific needs and preferences.

Furthermore, we will explore the exciting world of interactive visualizations, which can enhance the user experience by allowing for direct engagement with the data. Interactive visualizations can reveal additional insights and facilitate better understanding, as users can explore the data by interacting with the plot.

We will also discuss geospatial data visualization, an essential tool for displaying location-based information. By representing geospatial data visually, you can uncover geographic patterns and trends, enabling you to make better-informed decisions in various fields, such as urban planning, transportation, and environmental monitoring.

Our learning goals for this chapter are:

 * Learn to create custom data visualizations that cater to your specific needs and preferences.
 * Gain proficiency in creating interactive visualizations to enhance user engagement and understanding.
 * Understand the basics of geospatial data visualization and its applications in various fields.

## 9.1: Customizing Plots

Data visualization is an important part of data science. It plays a crucial role in data analysis by helping us understand patterns, trends, and relationships within the data. It also helps us comprehend complex data by presenting it in a visually appealing and easy-to-understand way. Python provides several libraries for data visualization, such as `matplotlib`, `seaborn`, Plotly, and many more. We will focus on customizing plots in the `matplotlib` and `seaborn` libraries, which are two of the most commonly used libraries for data visualization in Python.

Customizing plots is essential for creating effective and informative visualizations. It helps to highlight important trends and patterns in the data, and makes the visualization more appealing and understandable to the audience. Therefore, as a data scientist, it is important to have a good understanding of these customization options and how to use them effectively.

### `matplotlib`

`matplotlib` is widely used for scientific visualization and data exploration. `matplotlib` provides a wide range of customization options, such as changing the color, line style, marker style, label, title, and many more. These customization options can help you create high-quality visualizations that meet your specific needs.

We will explore some of the customization options available in `matplotlib`:

 1. Changing the Line Style: The line style can be customized using the `linestyle` parameter. The default line style is a solid line, but other options are `'None'`, `'solid'`, `'dashed'`, `'dashdot'` and `'dotted'`. For example, to create a dashed line, we can use the following code:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, linestyle='dashed')
plt.show()
```

 2. Changing the Color: The color of the line can be customized using the `color` parameter. The default color is blue, but other options include red, green, yellow, and many more. For example, to create a red line, we can use the following code:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, color='red')
plt.show()
```

 3. Adding Labels: Labels can be added to the x-axis, y-axis, and the plot title using the `xlabel()`, `ylabel()`, and `title()` functions::

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, color='red')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sine Curve')
plt.show()
```

 4. Changing the Marker Style: The marker style can be customized using the `marker` parameter. The default marker style is a solid circle (`'o'`), but other options include squares (`'s'`), triangles (`'^'`), pentagons (`'p'`), and many more. For example, to create a square marker, we can use the following code:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, marker='s')
plt.show()
```

 5. Adding a Legend: A legend can be added to the plot using the `legend()` function. The legend displays the label of each line in the plot. For example:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='Sine')
plt.plot(x, y2, label='Cosine')
plt.legend()
plt.show()
```

 6. Adding a grid: The `grid()` function is used to add grid lines to your plots, making it easier to read and interpret the data displayed on the graph. Grid lines can be added to either or both the x and y axes, and you can customize their appearance, such as the line style, color, and transparency.

Let's put all of these ideas together, and create a fully-customized plot using the `matplotlib` library:

```python
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
```

In summary, we have explored some of the customization options available in `matplotlib`. We learned how to change the line style, color, marker style, add labels, and a legend to our plot. By using these customization options, we can create high-quality visualizations that are tailored to our specific needs. It is important to note that there are many more customization options available in `matplotlib`, and you can refer to the official documentation online for a complete list of options.

### `seaborn`

`seaborn` is another popular data visualization library in Python, which is built on top of `matplotlib`. It provides a high-level interface for creating visually appealing and informative visualizations with minimal code. `seaborn` also provides additional customization options that are not available in `matplotlib`, such as built-in color palettes, automatic plot styling, and more.

Here are some examples of how to customize plots with `seaborn`:

 1. Changing the Color Palette: `seaborn` provides a wide range of built-in color palettes, which can be used to customize the colors of the plot. For example, to use the "coolwarm" color palette, we can use the following code:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': [2, 4, 5, 5, 4, 6], 'z': [0, 0, 1, 2, 1, 2]})
sns.lineplot(df, x='x', y='y', hue='z', palette='coolwarm')
plt.show()
```

 2. Changing the Line Style: `seaborn` provides additional line styles that are not available in `matplotlib`, such as the "dashdot" and "dotted" line styles. For example, to create a dash-dot line, we can use the following code:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

sns.lineplot(x=x, y=y, linestyle='dashdot')
plt.show()
```

 3. Changing the Marker Style: `seaborn` provides additional marker styles that are not available in `matplotlib`, such as the diamond (`'D'`) and hexagon (`'h'`) marker styles. For example, to create a diamond marker, we can use the following code:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

sns.lineplot(x=x, y=y, marker='D')
plt.show()
```

 4. Adding a Grid or Background: `seaborn` provides a convenient function called `set_style()` for adding a grid or setting the background style to the plot, which can make it easier to read the visualization. This can take one of the built-in values `'white'`, `'dark'`, `'whitegrid'`, `'darkgrid'` or `'ticks'`. For example:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

sns.set_style('darkgrid')
sns.lineplot(x=x, y=y)
plt.show()
```

Let's put some of these ideas into creating a customized scatterplot using the `seaborn` library. First, we import the required libraries and load the example dataset:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load the example dataset
data = sns.load_dataset('tips')
```

Now we will create a simple scatterplot and customize its appearance by changing point styles, colors, and sizes:

```python
# Create a scatterplot with customized point style, color, and size
sns.scatterplot(x='total_bill', y='tip', data=data, marker='D', s=100, color='purple', label='Tips')

# Customize plot appearance
plt.title('Customized Tips Scatterplot')
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.legend(loc='upper right')

# Show the plot
plt.show()
```

In summary, we have explored customizing plots using both the `matplotlib` and `seaborn` Python libraries. By personalizing your plots, you can make them more informative, visually appealing, and engaging. As you continue to work with data visualizations, you'll discover many more customization options available to help you create the perfect plot for your analysis.

## 9.2: Interactive Visualizations

Interactive visualizations enable users to explore data more effectively by allowing them to zoom, pan, and hover over points for additional information. This interactivity can lead to a deeper understanding of the underlying data. We'll explore two popular libraries for creating interactive visualizations in Python: Plotly and Bokeh.

### Plotly

Plotly is a powerful graphing library that creates interactive, web-based visualizations. It supports a variety of plot types, including scatter plots, line charts, bar charts, and more. Let's see how to create an interactive scatter plot using Plotly.

First, we need to install the Plotly library:

```bash
pip install plotly
```

Next, import the required libraries and create a sample dataset:

```python
import plotly.express as px
import pandas as pd

# Create a sample dataset
data = pd.DataFrame({'X': range(1, 11), 'Y': [i**2 for i in range(1, 11)]})
```

Now, create an interactive scatter plot:

```python
# Create an interactive scatter plot
fig = px.scatter(data, x='X', y='Y', title='Interactive Scatter Plot', labels={'X': 'X-axis', 'Y': 'Y-axis'})

# Show the plot in browser
fig.show()
```

### Bokeh

Bokeh is another popular library for creating interactive visualizations in Python. It provides an easy-to-use interface for generating web-based plots. Let's create an interactive line chart using Bokeh.

First, we need to install the Bokeh library:

```bash
pip install bokeh
```

Next, import the required libraries and generate a sample dataset:

```python
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import HoverTool
import numpy as np

# Generate a sample dataset
x = np.linspace(0, 10, 50)
y = np.sin(x)
```

Now, we can create an interactive line chart:

```python
# Create a Bokeh figure
p = figure(title='Interactive Line Chart', x_axis_label='X-axis', y_axis_label='Y-axis', tools='pan,wheel_zoom,reset,box_zoom')

# Add a line to the figure
p.line(x, y, legend_label='Sine Wave', line_width=2)

# Add hover tool
hover = HoverTool(tooltips=[('X', '$x{0.00}'), ('Y', '$y{0.00}')], mode='vline')
p.add_tools(hover)

# Show the plot in browser
show(p)
```

In summary, interactive plots offer a more engaging and informative way to explore data, enabling users themselves to dive deeper into the information and uncover hidden insights. As you continue to work with data visualization, consider incorporating interactive elements to enhance your data storytelling capabilities.

## 9.3: Geospatial Data Visualization

Geospatial data visualization involves creating maps and other visual representations of geographical data. These visualizations can reveal patterns and trends related to location, such as population density, climate variations, or traffic congestion. We'll explore two popular libraries for geospatial data visualization in Python: GeoPandas and Folium.

### GeoPandas

GeoPandas is an open-source library that extends the capabilities of `pandas` to handle geospatial data. It allows for easy manipulation and visualization of geospatial data in Python. Let's create a simple choropleth map using GeoPandas.

First, we need to install the GeoPandas library:

```bash
pip install geopandas
```

Next, import the required libraries and load a sample dataset:

```python
import geopandas as gpd
import matplotlib.pyplot as plt

# Load a sample dataset (GeoJSON format)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
```

Now, we can create a choropleth map showing population density:

```python
# Calculate population density
world['area_approx'] = world.area
world['pop_density'] = world['pop_est'] / world['area_approx']

# Create a choropleth map
ax = world.plot(column='pop_density', cmap='YlGn', legend=True, figsize=(15, 10))

# Customize the appearance
ax.set_title('World Population Density')
ax.set_axis_off()

# Show the plot
plt.show()
```

### Folium

Folium is a powerful library for creating interactive maps in Python. It builds on the Leaflet JavaScript library and allows for the creation of sophisticated maps with minimal code. Let's create an interactive map using Folium.

First, we need to install the Folium library:

```bash
pip install folium
```

Next, import the required libraries and create a base map:

```python
import folium

# Create a base map centered at a specific location
m = folium.Map(location=[45.523, -122.675], zoom_start=13)
```

Now, we can add markers to the map:

```python
# Add a marker
folium.Marker(
    location=[45.5244, -122.6699],
    popup='The Waterfront',
    icon=folium.Icon(color='green', icon='info-sign'),
).add_to(m)

# Add another marker
folium.Marker(
    location=[45.5215, -122.6261],
    popup='Mt. Tabor Park',
    icon=folium.Icon(color='red', icon='info-sign'),
).add_to(m)

# Save the map as local webpage, ready to load into browser
m.save('folium.html')
```

In summary, we have explored geospatial data visualization using GeoPandas and Folium. Geospatial visualizations can provide valuable insights into location-based data and help you uncover hidden patterns and trends. As you continue to work with data visualization, consider incorporating geospatial elements to enhance your data analysis capabilities.
