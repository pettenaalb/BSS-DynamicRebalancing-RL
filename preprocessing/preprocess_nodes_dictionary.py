import os
import osmnx as ox
import networkx as nx
import pickle

from tqdm import tqdm
from utils import nodes_within_radius

def initialize_graph(graph_path: str = None) -> nx.MultiDiGraph:
    if os.path.isfile(graph_path):
        print("Network file already exists. Loading the network data... ")
        graph = ox.load_graphml(graph_path)
        print("Network data loaded successfully.")
    else:
        # Raise an error if the graph file does not exist
        raise FileNotFoundError("Network file does not exist. Please check the file path.")

    return graph


def main():
    graph = initialize_graph('../data/utils/cambridge_network.graphml')

    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    nodes_dict = {node_id: (row['y'], row['x']) for node_id, row in nodes_gdf.iterrows()}

    radius = 100
    print("Creating nearby nodes dictionary...")
    nearby_nodes_dict = {node_id: nodes_within_radius(node_id, nodes_dict, radius) for node_id in tqdm(nodes_dict, desc="Nodes")}

    # Save dictionary to a file
    with open('../data/utils/nearby_nodes.pkl', 'wb') as file:
        pickle.dump(nearby_nodes_dict, file)



if __name__ == '__main__':
    main()