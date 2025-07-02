import osmnx as ox
import matplotlib.pyplot as plt

"""
This is a one time use script to generate the road network of interested city and area. More is the area, slower is the process.
For my use, I am using Munich city as an example.
"""

# Getting Munich city center coordinates
munich_point = ox.geocode("Munich, Germany")
# Create a graph within 8 km radius around city center
G = ox.graph_from_point(munich_point, dist=8000, network_type='drive')
# Visualisation of the map
fig, ax = ox.plot_graph(G,figsize=(40,40))
plt.show()
# saving the grpah in graphml format for further use
#ox.save_graphml(G, filepath="munich_drive.graphml")