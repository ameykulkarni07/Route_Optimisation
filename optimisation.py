from depot_delivery import MunichDeliveryNetwork
from tsp_ga import TSP_GA_Optimizer
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
from shapely.geometry import Point


def plot_route(graph, all_nodes, route_indices):
    """Visualize the route on the map using current OSMnx API"""
    fig, ax = ox.plot_graph(graph, show=False, close=False, node_size=0)

    # Plot all locations
    for i, node in enumerate(all_nodes):
        point = Point(graph.nodes[node]['x'], graph.nodes[node]['y'])
        color = 'red' if i == 0 else 'blue'
        label = 'Depot' if i == 0 else ('Deliveries' if i == 1 else None)
        ax.scatter(point.x, point.y, c=color, s=100 if i == 0 else 50, label=label)

    # Draw the route
    route_nodes = [all_nodes[i] for i in route_indices]
    for i in range(len(route_nodes) - 1):
        path = nx.shortest_path(graph, route_nodes[i], route_nodes[i + 1], weight='length')
        edge_nodes = list(zip(path[:-1], path[1:]))
        lines = []
        for u, v in edge_nodes:
            # Check if edge exists (some might be one-way)
            if not graph.has_edge(u, v):
                if graph.has_edge(v, u):  # Try reverse direction
                    u, v = v, u
                else:
                    continue

            # Get the edge geometry
            edge_data = graph.get_edge_data(u, v)
            for key, data in edge_data.items():
                if 'geometry' in data:
                    # Convert LineString to plot coordinates
                    xs, ys = data['geometry'].xy
                    ax.plot(xs, ys, 'r-', linewidth=2, alpha=0.7)

    plt.legend()
    plt.title("Optimized Delivery Route")
    plt.show()


if __name__ == "__main__":
    # 1. Setup delivery network
    delivery_network = MunichDeliveryNetwork()
    delivery_network.select_delivery_locations(num_locations=20)

    # 2. Get problem definition
    problem = delivery_network.get_problem_definition()

    # 3. Run GA optimization
    ga = TSP_GA_Optimizer(problem['distance_matrix'], problem['depot_index'])
    result = ga.optimize(
        population_size=150,
        generations=300,
        cx_prob=0.85,
        mut_prob=0.15
    )

    # 4. Display results
    print("\nOptimization Results:")
    print(f"Best route indices: {result['best_route']}")
    print(f"Total distance: {result['best_distance']:.2f} meters")

    # 5. Visualize the best route
    plot_route(delivery_network.G, problem['all_locations'], result['best_route'])