import os
import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np

from tqdm import tqdm
from utils import compute_distance, kahan_sum, nodes_within_radius

params = {
    'place': ["Cambridge, Massachusetts, USA"],
    'network_type': 'bike',

    'data_path': "../data/",
    'graph_file': "utils/cambridge_network.graphml",
    'year': 2022,
    'month': [9, 10],

    'day_of_week': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}

# ----------------------------------------------------------------------------------------------------------------------

def initialize_graph(graph_path: str = None) -> nx.MultiDiGraph:
    if os.path.isfile(graph_path):
        print("Network file already exists. Loading the network data... ")
        graph = ox.load_graphml(graph_path)
        print("Network data loaded successfully.")
    else:
        # Raise an error if the graph file does not exist
        raise FileNotFoundError("Network file does not exist. Please check the file path.")

    return graph

# ----------------------------------------------------------------------------------------------------------------------

def build_pmf_matrix(rate_matrix: pd.DataFrame, nearby_nodes_dict: dict[str, dict[str, tuple]], nodes_dict: dict[str, tuple]) -> pd.DataFrame:
    """
    Build the PMF matrix from the rate matrix.

    Parameters:
        - rate_matrix (pd.DataFrame): The rate matrix.
        - nearby_nodes_dict (dict): A dictionary containing the nearby nodes for each node.
        - nodes_dict (dict): A dictionary containing the coordinates of each node.

    Returns:
        - pmf_df (pd.DataFrame): The PMF matrix.
    """
    pmf_df = rate_matrix.copy(deep=True)

    non_zero_rows = pmf_df[pmf_df.sum(axis=1) != 0].index.astype(int)

    for row in non_zero_rows:
        # Extract row rates
        rates = pmf_df.loc[row]
        non_zero_nodes = rates[~rates.eq(0)].index.astype(int)
        zero_nodes = rates[rates.eq(0)].index.astype(int)

        for node_id in zero_nodes:
            nearby_nodes = nearby_nodes_dict[node_id]
            nearby_non_zero_nodes = {}
            for nz_id in non_zero_nodes:
                if nz_id in nearby_nodes:
                    nearby_non_zero_nodes[nz_id] = nearby_nodes[nz_id]

            if len(nearby_non_zero_nodes) != 0:
                coords = nodes_dict[node_id]
                distances = np.array([compute_distance(coords, nn_zn_coords) for nn_zn_coords in nearby_non_zero_nodes.values()])
                rts = np.array([rates.loc[nn_zn_id] for nn_zn_id in nearby_non_zero_nodes])
                num, den = 0, 0
                for distance, rate in zip(distances, rts):
                    num += rate/distance
                    den += 1/distance
                pmf_df.loc[row, node_id] = num/den

    zero_rows = pmf_df[pmf_df.sum(axis=1) == 0].index.astype(int)

    for idx in zero_rows:
        nearby_nodes = nearby_nodes_dict[idx]

        nearby_non_zero_nodes = {}
        for nz_id in non_zero_rows:
            if nz_id in nearby_nodes:
                nearby_non_zero_nodes[nz_id] = nearby_nodes[nz_id]

        if len(nearby_non_zero_nodes) != 0:
            coords = nodes_dict[idx]
            distances = np.array([compute_distance(coords, nn_zn_coords) for nn_zn_coords in nearby_non_zero_nodes.values()])
            num, den = pd.Series(0, index=pmf_df.columns), 0
            for distance, nn_zn_id in zip(distances, nearby_non_zero_nodes):
                num += pmf_df.loc[nn_zn_id]
                den += 1/distance
            pmf_df.loc[idx] = num/den

    return pmf_df


def build_pmf_matrix_external_trips(df: pd.DataFrame, nearby_nodes_dict: dict[str, dict[str, tuple]], nodes_dict: dict[str, tuple]) -> pd.DataFrame:
    """
    Build the PMF matrix from the rate matrix.

    Parameters:
        - rate_matrix (pd.DataFrame): The rate matrix.
        - nearby_nodes_dict (dict): A dictionary containing the nearby nodes for each node.
        - nodes_dict (dict): A dictionary containing the coordinates of each node.

    Returns:
        - pmf_df (pd.DataFrame): The PMF matrix.
    """

    non_zero_nodes = df[~df.eq(0)].index.astype(int)
    zero_nodes = df[df.eq(0)].index.astype(int)

    if 10000 in non_zero_nodes:
        non_zero_nodes = non_zero_nodes[non_zero_nodes != 10000]

    if 10000 in zero_nodes:
        zero_nodes = zero_nodes[zero_nodes != 10000]

    for node_id in zero_nodes:
        nearby_nodes = nearby_nodes_dict[node_id]
        nearby_non_zero_nodes = {}
        for nz_id in non_zero_nodes:
            if nz_id in nearby_nodes:
                nearby_non_zero_nodes[nz_id] = nearby_nodes[nz_id]

        if len(nearby_non_zero_nodes) != 0:
            coords = nodes_dict[node_id]
            distances = np.array([compute_distance(coords, nn_zn_coords) for nn_zn_coords in nearby_non_zero_nodes.values()])
            rts = np.array([df.loc[nn_zn_id] for nn_zn_id in nearby_non_zero_nodes])
            num, den = 0, 0
            for distance, rate in zip(distances, rts):
                num += rate/distance
                den += 1/distance
            df.loc[node_id] = num/den

    return df

# ----------------------------------------------------------------------------------------------------------------------

def main():
    # Initialize the graph
    print("Initializing the graph... ")
    graph = initialize_graph(params['data_path'] + params['graph_file'])

    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    nodes_dict = {node_id: (row['y'], row['x']) for node_id, row in nodes_gdf.iterrows()}

    radius = 500
    nearby_nodes_dict = {node_id: nodes_within_radius(node_id, nodes_dict, radius) for node_id in tqdm(nodes_dict, desc="Building Nearby Nodes", dynamic_ncols=True)}

    tbar = tqdm(total=len(params['day_of_week']) * 8, desc="Processing Data", position=0, dynamic_ncols=True)

    print(f"\nBuilding the PMF matrices... ", end=" ")

    for day in params['day_of_week']:
        for timeslot in range(0, 8):
            mon_str = str(params['month'][0]).zfill(2) + '-' + str(params['month'][-1]).zfill(2)
            matrix_path = params['data_path'] + 'matrices/' + mon_str + '/' + str(timeslot).zfill(2) + '/'
            rate_matrix = pd.read_csv(matrix_path + day.lower() + '-rate-matrix.csv', index_col='osmid')

            rate_matrix.index = rate_matrix.index.astype(int)
            rate_matrix.columns = rate_matrix.columns.astype(int)

            saved_row = rate_matrix.loc[10000, :].copy()
            saved_col = rate_matrix.loc[:, 10000].copy()

            rate_matrix = rate_matrix.drop(index=10000, columns=10000)

            # Build PMF matrix
            pmf_matrix = build_pmf_matrix(rate_matrix, nearby_nodes_dict, nodes_dict)
            saved_row = build_pmf_matrix_external_trips(saved_row, nearby_nodes_dict, nodes_dict)
            saved_col = build_pmf_matrix_external_trips(saved_col, nearby_nodes_dict, nodes_dict)

            pmf_matrix.loc[10000, :] = saved_row
            pmf_matrix.loc[:, 10000] = saved_col

            total_sum = kahan_sum(pmf_matrix.to_numpy().flatten())
            pmf_matrix = pmf_matrix / total_sum

            # Save the interpolated rate matrix to a CSV file
            pmf_matrix.to_csv(matrix_path + day.lower() + '-pmf-matrix.csv', index=True)

            tbar.update(1)


if __name__ == '__main__':
    main()