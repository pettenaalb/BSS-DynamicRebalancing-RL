from operator import index

import osmnx as ox
import pandas as pd
import os
import networkx as nx
from simulator.utils import plot_graph_with_colored_nodes, kahan_sum
import numpy as np
from tqdm import tqdm
import logging
from utils import haversine

params = {
    'place': ["Cambridge, Massachusetts, USA"],
    'network_type': 'drive',

    'data_path': "../data/",
    'graph_file': "cambridge_network.graphml",
    'year': 2022,
    'month': 1,

    'time_interval': 3600*24   # 1 hour
}


def initialize_graph(graph_path: str = None) -> nx.MultiDiGraph:
    if os.path.isfile(graph_path):
        print("Network file already exists. Loading the network data... ")
        graph = ox.load_graphml(graph_path)
        print("Network data loaded successfully.")
    else:
        # Raise an error if the graph file does not exist
        raise FileNotFoundError("Network file does not exist. Please check the file path.")

    return graph


def nodes_within_radius(target_node: str, nodes_dict: dict[str, tuple], radius: int) -> dict[str, tuple]:
    # Get coordinates of the target node
    target_coords = nodes_dict.get(target_node)
    if not target_coords:
        raise ValueError("Target node not found in nodes dictionary")

    # Find all nodes within the radius and return as a dictionary
    nearby_nodes = {
        node_id: coords for node_id, coords in nodes_dict.items()
        if node_id != target_node and haversine(target_coords, coords) <= radius
    }

    return nearby_nodes


def interpolate_nodes(rate_matrix: pd.DataFrame, nearby_nodes_dict: dict[str, dict[str, tuple]], nodes_dict: dict[str, tuple]) -> pd.DataFrame:
    rate_df = rate_matrix.copy(deep=True)

    non_zero_indices = rate_df[rate_df.sum(axis=1) != 0].index.astype(int)

    tbar = tqdm(total=rate_df.shape[0], desc="Interpolating nodes")

    for idx in non_zero_indices:
        # Extract row rates
        rates = rate_df.loc[idx]
        non_zero_ids = rates[~rates.eq(0)].index.astype(int)
        zero_ids = rates[rates.eq(0)].index.astype(int)

        for node_id in zero_ids:
            nearby_nodes = nearby_nodes_dict[node_id]

            nearby_non_zero_nodes = {}
            for nz_id in non_zero_ids:
                if nz_id in nearby_nodes:
                    nearby_non_zero_nodes[nz_id] = nearby_nodes[nz_id]

            if len(nearby_non_zero_nodes) != 0:
                coords = nodes_dict[node_id]
                distances = np.array([haversine(coords, nn_zn_coords) for nn_zn_coords in nearby_non_zero_nodes.values()])
                rts = np.array([rates.loc[str(nn_zn_id)] for nn_zn_id in nearby_non_zero_nodes])
                num, den = 0, 0
                for distance, rate in zip(distances, rts):
                    num += rate/distance
                    den += 1/distance
                rate_df.loc[idx, str(node_id)] = num/den

        tbar.update(1)

    zero_indices = rate_df[rate_df.sum(axis=1) == 0].index.astype(int)

    for idx in zero_indices:
        nearby_nodes = nearby_nodes_dict[idx]

        nearby_non_zero_nodes = {}
        for nz_id in non_zero_indices:
            if nz_id in nearby_nodes:
                nearby_non_zero_nodes[nz_id] = nearby_nodes[nz_id]

        if len(nearby_non_zero_nodes) != 0:
            coords = nodes_dict[idx]
            distances = np.array([haversine(coords, nn_zn_coords) for nn_zn_coords in nearby_non_zero_nodes.values()])
            num, den = pd.Series(0, index=rate_df.columns), 0
            for distance, nn_zn_id in zip(distances, nearby_non_zero_nodes):
                num += rate_df.loc[nn_zn_id]
                den += 1/distance
            rate_df.loc[idx] = num/den
        tbar.update(1)

    return rate_df


def main():
    ox.settings.use_cache = True
    logging.basicConfig(level=logging.INFO, filemode='w', filename='dummy_output.log')

    # Initialize the graph
    print("Initializing the graph... ")
    graph = initialize_graph(params['data_path'] + params['graph_file'])

    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    nodes_dict = {node_id: (row['y'], row['x']) for node_id, row in nodes_gdf.iterrows()}

    nearby_nodes_dict = {}
    radius = 1000

    for node_id in nodes_dict:
        nearby_nodes_dict[node_id] = nodes_within_radius(node_id, nodes_dict, radius)

    # for node_id in nearby_nodes_dict:
    #     logging.info(f"\nNode {node_id} has {nearby_nodes_dict[node_id]}")

    rate_matrix = pd.read_csv("../data/matrices/09-10/02/monday-rate-matrix.csv", index_col=0)

    rate_matrix = rate_matrix.drop(index=10000, columns='10000')

    interpolated_rate_matrix = interpolate_nodes(rate_matrix, nearby_nodes_dict, nodes_dict)

    interpolated_rate_matrix.to_csv('interpolated-rate-matrix.csv')

    total_sum = kahan_sum(interpolated_rate_matrix.to_numpy().flatten())
    interpolated_rate_matrix = interpolated_rate_matrix / total_sum

    print(kahan_sum(interpolated_rate_matrix.to_numpy().flatten()))

    interpolated_rate_matrix.to_csv('scaled-rate-matrix.csv')


if __name__ == '__main__':
    main()
