import geopandas as gpd
import matplotlib.pyplot as plt

# Load a sample dataset (GeoJSON format)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
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
