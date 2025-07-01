import osmnx as ox
import numpy as np
import networkx as nx
import random


class MunichDeliveryNetwork:
    def __init__(self, graph_file="munich_drive.graphml"):
        """
        Initialize the Munich delivery network with a saved graph file.
        Handles graph loading and distance matrix generation only.

        Args:
            graph_file: Path to the saved OSMnx graph file
        """
        # Load and prepare the graph
        self.G = ox.load_graphml(graph_file)
        self.G = self.G.to_undirected()  # Convert to undirected

        # Get Munich coordinates and center node
        self.munich_point = ox.geocode("Munich, Germany")
        self.center_node = ox.distance.nearest_nodes(self.G, self.munich_point[1], self.munich_point[0])
        # Initialize problem parameters
        self.depot = self.center_node
        self.delivery_locations = []
        self.all_locations = []
        self.distance_matrix = None

    def select_delivery_locations(self, num_locations=10, min_distance=1000, max_distance=7500):
        """
        Select random delivery locations within distance range from depot.
        Automatically generates the distance matrix after selection.

        Args:
            num_locations: Number of delivery locations to select
            min_distance: Minimum distance from depot (meters)
            max_distance: Maximum distance from depot (meters)
        """
        all_nodes = list(self.G.nodes())
        all_nodes.remove(self.depot)

        # Calculate distances from depot
        lengths = nx.single_source_dijkstra_path_length(self.G, self.depot, weight='length')

        # Filter nodes within distance range
        candidate_nodes = [
            node for node in all_nodes
            if min_distance <= lengths.get(node, float('inf')) <= max_distance
        ]

        # Adjust if not enough candidates
        if len(candidate_nodes) < num_locations:
            print(f"Warning: Only {len(candidate_nodes)} nodes meet distance criteria")
            num_locations = len(candidate_nodes)

        self.delivery_locations = random.sample(candidate_nodes, num_locations)
        self.all_locations = [self.depot] + self.delivery_locations
        self._create_distance_matrix()

    def _create_distance_matrix(self):
        """Internal method to create distance matrix between all locations"""
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

        Returns:
            dict: {
                'distance_matrix': numpy array of distances,
                'num_locations': total locations (depot + deliveries),
                'depot_index': index of depot (always 0),
                'all_locations': list of node IDs in order
            }
        """
        return {
            'distance_matrix': self.distance_matrix,
            'num_locations': len(self.all_locations),
            'depot_index': 0,
            'all_locations': self.all_locations
        }