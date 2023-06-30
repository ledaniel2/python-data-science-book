from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import HoverTool
import numpy as np

# Generate a sample dataset
x = np.linspace(0, 10, 50)
y = np.sin(x)
# Create a Bokeh figure
p = figure(title='Interactive Line Chart', x_axis_label='X-axis', y_axis_label='Y-axis', tools='pan,wheel_zoom,reset,box_zoom')

# Add a line to the figure
p.line(x, y, legend_label='Sine Wave', line_width=2)

# Add hover tool
hover = HoverTool(tooltips=[('X', '$x{0.00}'), ('Y', '$y{0.00}')], mode='vline')
p.add_tools(hover)

# Show the plot in browser
show(p)
