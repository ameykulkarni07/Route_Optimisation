import osmnx as ox
import numpy as np
import networkx as nx
import random

class MunichDeliveryNetwork:
    def __init__(self, graph_file="munich_drive.graphml"):
        """
        Initialize the Munich delivery network with a graph file. This is the file which we save in the road_network.py file.
        Reasonable Assumption 1 :
        For simplicity of the problem, we convert the road network to undirected. This means we can travel in both direactions of the road.
        """
        # Load and prepare the graph
        self.G = ox.load_graphml(graph_file)
        self.G = self.G.to_undirected()  # This function converts the G map to undirected case.
        # Get Munich coordinates and center node as the starting point
        self.munich_point = ox.geocode("Munich, Germany")
        # We find the nearest node to the centre of city. munich_point is a tuple of (latitude, longitude). We assign
        # the nearest node as the depot in the next line
        self.center_node = ox.distance.nearest_nodes(self.G, self.munich_point[1], self.munich_point[0])
        self.depot = self.center_node
        # Initialize empty dataframe for delivery locations and distance matrix
        self.delivery_locations = []
        self.all_locations = []
        self.distance_matrix = None

    def select_delivery_locations(self, num_locations=10, min_distance=1000, max_distance=7500):
        """
        Select random delivery locations within distance range from depot.
        Automatically generates the distance matrix after selection.
        User can select the number of locations and the min max distance from the depot.
        """
        all_nodes = list(self.G.nodes())
        all_nodes.remove(self.depot)  # Remove depot from candidate nodes
        # Calculate distances from depot
        lengths = nx.single_source_dijkstra_path_length(self.G, self.depot, weight='length')
        # Filter nodes within distance range. We get the distance from depot to all other nodes.
        candidate_nodes = [node for node in all_nodes
            if min_distance <= lengths.get(node, float('inf')) <= max_distance]
        # Adjust if not enough candidates. Very unlikely.
        if len(candidate_nodes) < num_locations:
            print(f"Warning: Only {len(candidate_nodes)} nodes meet distance criteria")
            num_locations = len(candidate_nodes)
        # Randomly select delivery locations from candidates.
        self.delivery_locations = random.sample(candidate_nodes, num_locations)
        self.all_locations = [self.depot] + self.delivery_locations
        self._create_distance_matrix()

    def _create_distance_matrix(self):
        # Create a distance matrix for all locations including depot and deliveries. More information in the Documentation file.
        n = len(self.all_locations)
        self.distance_matrix = np.zeros((n, n))
        # Calculate shortest path distances between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                path_length = nx.shortest_path_length(
                    self.G, self.all_locations[i], self.all_locations[j], weight='length'
                )
                self.distance_matrix[i][j] = path_length
                self.distance_matrix[j][i] = path_length

    def get_problem_definition(self):
        """
        Return the complete problem definition for optimization.
        """
        return {
            'distance_matrix': self.distance_matrix,
            'num_locations': len(self.all_locations),
            'depot_index': 0,
            'all_locations': self.all_locations
        }