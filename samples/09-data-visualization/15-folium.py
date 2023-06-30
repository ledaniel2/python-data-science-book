import folium

# Create a base map centered at a specific location
m = folium.Map(location=[45.523, -122.675], zoom_start=13)

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
print('Saved file: folium.html')
