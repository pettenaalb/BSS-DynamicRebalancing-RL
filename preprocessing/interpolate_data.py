import os
from operator import index

import GPy
import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
import warnings

from tqdm import tqdm
from simulator.utils import kahan_sum, plot_graph_with_colored_nodes

params = {
    'place': ["Cambridge, Massachusetts, USA"],
    'network_type': 'bike',

    'data_path': "../data/",
    'graph_file': "cambridge_network.graphml",
    'year': 2022,
    'month': [9, 10],

    'day_of_week': ["Monday"],
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

# ----------------------------------------------------------------------------------------------------------------------

def interpolate_rates_with_gaussian_process_per_axis(rate_matrix: pd.DataFrame, coordinates: dict[int, tuple], axis: int) -> pd.DataFrame:
    """
    Interpolate the rates using Gaussian Process Regression along specified axis.

    Parameters:
        - rate_matrix (pd.DataFrame): The rate matrix containing the rates.
        - coordinates (np.array): Array of node coordinates.
        - axis (int): The axis along which to interpolate (0 for rows, 1 for columns).

    Returns:
        - pd.DataFrame: The rate matrix with interpolated values.
    """
    rate_df = rate_matrix.copy(deep=True)
    coords_to_predict = np.array([coordinates[int(node_id)] for node_id in (rate_df.columns.tolist() if axis == 0 else rate_df.index.tolist())])

    tbar = tqdm(total=rate_df.shape[axis], desc="Interpolating Rates", position=0, leave=True)

    for idx in (rate_df.index if axis == 0 else rate_df.columns):
        # Extracting row or column rates depending on axis
        rates = rate_df.loc[idx] if axis == 0 else rate_df.loc[:, idx]
        non_zero_ids = rates[~rates.eq(0)].index
        non_zero_rates = rates[~rates.eq(0)].values

        if len(non_zero_rates) > 0:
            # Rescale for better numerical stability in GP
            r_min, r_max = non_zero_rates.min(), non_zero_rates.max()
            scaling_factor = 1 if r_min == 0 else 10 ** -np.floor(np.log10(r_min))
            non_zero_rates = non_zero_rates * scaling_factor

            # Training data for GP
            X_train = np.array([coordinates[int(node_id)] for node_id in non_zero_ids])
            y_train = non_zero_rates.reshape(-1, 1)

            # Gaussian Process model
            kernel = GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=0.001)
            gp_model = GPy.models.GPRegression(X_train, y_train, kernel)
            gp_model.optimize()

            # Make predictions for the entire range of nodes
            predicted_rates, _ = gp_model.predict(coords_to_predict)

            # Rescale the predicted rates back to original scale
            predicted_rates = predicted_rates / scaling_factor

            # Saving predicted coordinates and predicted rates in the dataframe
            for i, node_id in enumerate(rate_df.columns if axis == 0 else rate_df.index):
                value = predicted_rates[i] if predicted_rates[i] > 0 else 0
                if value < 1e-6:
                    value = 0
                if axis == 0:
                    rate_df.loc[int(idx), node_id] = value
                else:
                    rate_df.loc[node_id, str(idx)] = value

        tbar.update(1)

    tbar.close()
    return rate_df


def rescale_interpolated_rates(rate_matrix: pd.DataFrame, interpolated_rate_matrix) -> pd.DataFrame:
    """
    Rescale the interpolated rates to match the original rates.

    Parameters:
        - rate_matrix (pd.DataFrame): The original rate matrix.
        - interpolated_rate_matrix (pd.DataFrame): The interpolated rate matrix.

    Returns:
        - pd.DataFrame: The rescaled interpolated rate matrix
    """
    df = rate_matrix.copy()
    int_df = interpolated_rate_matrix.copy()

    for idx in int_df.index:
        mean = int_df.loc[idx].mean()
        int_df.loc[idx] = int_df.loc[idx] / mean

    # Kahan summation to avoid floating-point errors
    total_rate = kahan_sum(df.to_numpy().flatten())
    total_int_rate = kahan_sum(int_df.to_numpy().flatten())

    # Separate operations to increase precision
    div = total_rate / total_int_rate
    int_df = int_df * div

    return int_df

# ----------------------------------------------------------------------------------------------------------------------

def interpolate_rates(rate_matrix: pd.DataFrame, coordinates: dict[int, tuple]) -> pd.DataFrame:
    rate_df = rate_matrix.copy(deep=True)



    pass

# ----------------------------------------------------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in square")

    # Initialize the graph
    print("Initializing the graph... ")
    graph = initialize_graph(params['data_path'] + params['graph_file'])

    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    coordinates_dict = {node_id: (row['y'], row['x']) for node_id, row in nodes_gdf.iterrows()}

    # tbar = tqdm(total=len(params['day_of_week']) * 8, desc="Processing Data")

    for day in params['day_of_week']:
        for timeslot in range(2, 3):
            # Load the rate matrix for the specified day and timeslot
            mon_str = str(params['month'][0]).zfill(2) + '-' + str(params['month'][-1]).zfill(2)
            matrix_path = params['data_path'] + 'matrices/' + mon_str + '/' + str(timeslot).zfill(2) + '/'
            rate_matrix = pd.read_csv(matrix_path + day.lower() + '-rate-matrix.csv', index_col='osmid')

            # row_10000 = rate_matrix.loc[[10000]]
            # row_10000 = row_10000.drop(columns='10000')
            # col_10000 = rate_matrix[['10000']]
            # col_10000 = col_10000.drop(index=10000)

            rate_matrix = rate_matrix.drop(index=10000, columns='10000')

            # Interpolate missing rates using Gaussian Process
            print("Interpolating missing rates using Gaussian Process... ")
            interpolated_rate_matrix = interpolate_rates_with_gaussian_process_per_axis(rate_matrix, coordinates_dict, axis=1)
            interpolated_rate_matrix = interpolate_rates_with_gaussian_process_per_axis(interpolated_rate_matrix, coordinates_dict, axis=0)

            # Save the interpolated rate matrix to a CSV file
            print("Saving the interpolated rate matrix to a CSV file... ")
            interpolated_rate_matrix.to_csv(matrix_path + day.lower() + '-interpolated-rate-matrix.csv', index=True)

            # row_10000_int = interpolate_rates_with_gaussian_process_per_axis(row_10000, coordinates_dict, axis=0)
            # col_10000_int = interpolate_rates_with_gaussian_process_per_axis(col_10000, coordinates_dict, axis=1)
            #
            # interpolated_rate_matrix[10000] = col_10000_int
            # interpolated_rate_matrix.loc[10000] = row_10000_int.loc[10000]

            # # Rescale the interpolated rates to match the original rates
            # print("Rescaling the interpolated rates to match the original rates... ")
            # interpolated_rate_matrix = rescale_interpolated_rates(rate_matrix, interpolated_rate_matrix)
            #
            # # Save the interpolated rate matrix to a CSV file
            # print("Saving the interpolated rate matrix to a CSV file... ")
            # interpolated_rate_matrix.to_csv(matrix_path + day.lower() + '-rescaled-interpolated-rate-matrix.csv', index=True)

            # tbar.update(1)


if __name__ == '__main__':
    main()