import os
import pandas as pd
import osmnx as ox
import networkx as nx

from geopy.distance import great_circle
from tqdm import tqdm

params = {
    'place': ["Cambridge, Massachusetts, USA"],
    'network_type': 'bike',

    'data_path': "data/",
    'graph_file': "cambridge_network.graphml",
    'year': 2022,
    'month': [1],

    'time_duration': [31*24*3600]
}


def find_nearby_nodes(graph: nx.MultiDiGraph, target_node: int, radius_meters: float) -> list[int]:
    """
    Find nearby nodes within a specified radius around the target node.

    Parameters:
        - graph (nx.MultiDiGraph): The graph representing the road network.
        - target_node (int): The target node to find nearby nodes around.
        - radius_meters (float): The radius in meters within which to find nearby nodes.

    Returns:
        - list: A list of node IDs that are within the specified radius around the target
    """
    # Check if the target node is in the graph
    if target_node not in graph:
        raise ValueError(f"Node {target_node} is not in the graph.")

    target_coords = (graph.nodes[target_node]['y'], graph.nodes[target_node]['x'])
    nearby_nodes = []

    # Find nearby nodes within the specified radius
    for node in graph.nodes:
        if node != target_node:
            node_coords = (graph.nodes[node]['y'], graph.nodes[node]['x'])
            distance = great_circle(target_coords, node_coords).meters

            if distance <= radius_meters:
                nearby_nodes.append(node)

    return nearby_nodes


def connect_disconnected_neighbors(graph: nx.MultiDiGraph, radius_meters: int):
    """
    Connect disconnected nodes in the graph by adding edges between them.

    Parameters:
        - graph (nx.MultiDiGraph): The graph representing the road network.
        - radius_meters (int): The radius in meters within which to connect disconnected nodes.
    """

    tbar = tqdm(total=len(graph.nodes), desc="Connecting disconnected nodes")

    for node in graph.nodes:
        # Check if the node has valid coordinates
        if 'y' not in graph.nodes[node] or 'x' not in graph.nodes[node]:
            print(f"Node {node} does not have valid coordinates.")
            continue

        # Find nearby nodes within the specified radius
        nearby_nodes = find_nearby_nodes(graph, node, radius_meters)

        # Add an edge between the node and its neighbors if they are not already connected
        for neighbor in nearby_nodes:
            if not nx.has_path(graph, node, neighbor):
                node_coords = (graph.nodes[node]['y'], graph.nodes[node]['x'])
                neighbor_coords = (graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x'])

                if 'y' not in graph.nodes[neighbor] or 'x' not in graph.nodes[neighbor]:
                    print(f"Neighbor {neighbor} does not have valid coordinates.")
                    continue
                distance_meters = great_circle(node_coords, neighbor_coords).meters

                speed_kph = 15.0
                travel_time_hours = distance_meters / 1000 / speed_kph
                weight = travel_time_hours * 3600
                graph.add_edge(node, neighbor, length=distance_meters, speed_kph=speed_kph, weight=weight)

        tbar.update(1)


def initialize_graph(places: [str], network_type: str, graph_path: str = None, simplify_network: bool = False,
                     remove_isolated_nodes: bool = False) -> nx.MultiDiGraph:
    """
    Initialize the graph representing the road network.

    Parameters:
        - places (list): List of places to download the road network data.
        - network_type (str): Type of network to download.
        - graph_path (str): Path to save the downloaded graph data.
        - simplify_network (bool): Whether to simplify the network by consolidating intersections.
        - remove_isolated_nodes (bool): Whether to remove isolated nodes from the network.

    Returns:
        - nx.MultiDiGraph: The graph representing the road network.
    """
    if os.path.isfile(graph_path):
        # Load the network data if the file already exists
        print("Network file already exists. Loading the network data... ")
        graph = ox.load_graphml(graph_path)
        print("Network data loaded successfully.")
    else:
        # Download the network data if the file does not exist
        print("Network file does not exist. Downloading the network data... ")
        graph = ox.graph_from_place(places[0], network_type=network_type)
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)

        # Download the network data for additional places if specified and compose the graphs
        if len(places) > 1:
            for index in range(1, len(places)):
                grp = ox.graph_from_place(places[index], network_type=network_type)
                grp = ox.add_edge_speeds(grp)
                grp = ox.add_edge_travel_times(grp)
                graph = nx.compose(graph, grp)
                connect_disconnected_neighbors(graph, radius_meters=100)

        # Simplify the graph by consolidating intersections
        if simplify_network:
            G_proj = ox.project_graph(graph)
            G_cons = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=True)
            graph = ox.project_graph(G_cons, to_crs='epsg:4326')

        # Remove isolated nodes
        if remove_isolated_nodes:
            graph.remove_nodes_from(list(nx.isolates(graph)))

        ox.save_graphml(graph, graph_path)
        print("Network data downloaded and saved successfully.")

    return graph


def initialize_distance_matrix(G: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Initialize a distance matrix based on the graph.

    Parameters:
        - G (nx.MultiDiGraph): The graph representing the road network.

    Returns:
        - pd.DataFrame: The distance matrix for the graph.
    """

    # Get node IDs and create an empty square DataFrame
    node_ids = ox.graph_to_gdfs(G, edges=False).index
    df = pd.DataFrame(index=node_ids, columns=node_ids, dtype=int)

    # Initialize values to zero
    df = df.fillna(0)

    G_undirected = G.to_undirected()
    for i in tqdm(node_ids, desc="Processing Distances"):
        for j in node_ids:
            if i != j and j > i:
                distance = int(nx.shortest_path_length(G_undirected, i, j, weight='length'))
                df.at[i, j] = distance
                df.at[j, i] = distance

    return df


def main():
    # Initialize the graph
    print("Initializing the graph... ")
    graph = initialize_graph(params['place'], params['network_type'], params['data_path'] + params['graph_file'],
                             remove_isolated_nodes=True, simplify_network=True)

    if not os.path.isfile(params['data_path'] + 'distance-matrix.csv'):
        # Initialize the distance matrix
        print("Initializing the distance matrix... ")
        distance_matrix = initialize_distance_matrix(graph)

        # Save the distance matrix to a CSV file
        print("Saving the distance matrix to a CSV file... ")
        distance_matrix.to_csv(params['data_path'] + 'distance-matrix.csv', index=True)


if __name__ == '__main__':
    main()