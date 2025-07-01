import osmnx as ox
import matplotlib.pyplot as plt

# Get Munich city center coordinates
munich_point = ox.geocode("Munich, Germany")  # returns (lat, lon)

# Create a graph within 8 km radius around Munich city center
G = ox.graph_from_point(munich_point, dist=8000, network_type='drive')

# Plot the graph
fig, ax = ox.plot_graph(G,figsize=(40,40))
plt.show()

#ox.save_graphml(G, filepath="munich_drive.graphml")