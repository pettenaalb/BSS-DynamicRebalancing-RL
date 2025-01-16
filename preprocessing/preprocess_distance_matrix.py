import os
import platform
import pandas as pd
import osmnx as ox
import networkx as nx

from tqdm import tqdm

params = {
    'place': ["Cambridge, Massachusetts, USA"],
    'network_type': 'bike',

    'data_path': "../data/",
    'graph_file': "utils/cambridge_network.graphml",
    'year': 2022,
    'month': [1],

    'time_duration': [31*24*3600]
}

if platform.system() == "Linux":
    params['data_path'] = "/mnt/mydisk/edoardo_scarpel/data/"

def initialize_graph(graph_path: str = None) -> nx.MultiDiGraph:
    if os.path.isfile(graph_path):
        print("Network file already exists. Loading the network data... ")
        graph = ox.load_graphml(graph_path)
        print("Network data loaded successfully.")
    else:
        # Raise an error if the graph file does not exist
        raise FileNotFoundError("Network file does not exist. Please check the file path.")

    return graph


def initialize_distance_matrix(G: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Initialize a distance matrix based on the graph.

    Parameters:
        - G (nx.MultiDiGraph): The graph representing the road network.

    Returns:
        - pd.DataFrame: The distance matrix for the graph.
    """

    print("Calculating all the shortest paths... ")
    # Get node IDs and create an empty square DataFrame
    node_ids = ox.graph_to_gdfs(G, edges=False).index
    df = pd.DataFrame(index=node_ids, columns=node_ids, dtype='int')

    # Initialize values to zero
    df = df.fillna(0)

    # Calculate the shortest paths between all pairs of nodes
    G_undirected = G.to_undirected()
    distances = dict(nx.all_pairs_dijkstra_path_length(G_undirected, weight="length"))

    for i in node_ids:
        for j in node_ids:
            df.at[i, j] = int(distances[i][j])

    return df


def main():
    if not os.path.exists(params['data_path'] + 'utils/'):
        os.makedirs(params['data_path'] + 'utils/')
        print("Directory created: " + params['data_path'] + 'utils/')

    # Initialize the graph
    print("Initializing the graph... ")
    graph = initialize_graph(params['data_path'] + params['graph_file'])

    # Initialize the distance matrix
    print("Initializing the distance matrix... ")
    distance_matrix = initialize_distance_matrix(graph)

    # Save the distance matrix to a CSV file
    print("Saving the distance matrix to a CSV file... ")
    distance_matrix.to_csv(params['data_path'] + 'utils/distance_matrix.csv', index=True)


if __name__ == '__main__':
    main()