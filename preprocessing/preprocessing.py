import argparse
import os
import shutil
import pickle

import pandas as pd
import osmnx as ox

from .utils import *
from tqdm import tqdm

params = {
    'place': ["Cambridge, Massachusetts, USA"],
    'network_type': 'bike',

    'data_path': "data/",
    'graph_file': "utils/cambridge_network.graphml",
    'cell_data_path': "utils/cell_data.pkl",
    'global_rates_path': "utils/global_rates.pkl",
    'distance_matrix_path': "utils/distance_matrix.csv",
    'nearby_nodes_path': "utils/nearby_nodes.pkl",

    'year': 2022,
    'month': [9, 10],
    'nodes_to_remove': [(42.365455, -71.14254)],

    'day_of_week': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}

def main():
    # ---------- DOWNLOAD TRIP DATA ----------
    print("Starting preprocessing... Downloading trip data.")
    save_path = params['data_path'] + 'trips/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")

    tbar = tqdm(range(12), desc='Downloading files', position=0, leave=True)

    for month in range(0, 12):
        if not os.path.exists(save_path + str(params['year']) + str(month + 1).zfill(2) + '-bluebikes-tripdata.csv'):
            url = 'https://s3.amazonaws.com/hubway-data/2022' + str(month + 1).zfill(2) + '-bluebikes-tripdata.zip'
            download_and_extract(url, save_path, tbar)
        tbar.update(1)

    # if os.path.exists(save_path + '__MACOSX'):
    #     shutil.rmtree(save_path + '__MACOSX')

    # ---------- PREPROCESS TRIP DATA ----------
    print("Finished downloading trip data... Preprocessing trip data.")

    # Initialize the graph
    if not os.path.exists(params['data_path'] + 'utils/'):
        os.makedirs(params['data_path'] + 'utils/')
        print(f"Directory '{params['data_path'] + 'utils/'}' created.")

    print("Initializing the graph.")

    bbox = (42.3679, 42.3523, -71.0883, -71.1046)
    graph = initialize_graph(params['place'], params['network_type'], params['data_path'] + params['graph_file'],
                             remove_isolated_nodes=True, simplify_network=True,
                             nodes_to_remove=params['nodes_to_remove'], bbox=bbox)

    print(f'\nProcessing data for year {params["year"]} and month {params["month"]}...')
    trip_df = pd.DataFrame()
    for month in params['month']:
        path = params['data_path'] + 'trips/' + str(params['year']) + str(month).zfill(2) + '-bluebikes-tripdata.csv'
        if os.path.isfile(path):
            trip_df = pd.concat([trip_df, pd.read_csv(path)], ignore_index=True)
        else:
            print(f"Trip data file for month {month} does not exist. Skipping...")

    # Load the filtered stations
    filtered_stations = pd.read_csv(params['data_path'] + 'utils/filtered_stations.csv')

    tbar = tqdm(total=len(params['day_of_week']) * 8, desc="Processing Data", position=0, dynamic_ncols=True,
                leave=True)

    global_rates = {}
    for day in params['day_of_week']:
        for timeslot in range(0, 8):
            # print(f"Processing data for {day} - Time Slot {timeslot}...")
            # Compute the rates for each station pair
            # print("Computing Poisson rates... ")
            poisson_rates_df = compute_poisson_rates(trip_df, params['year'], params['month'], day, timeslot)

            # Transform the trip data to match the graph
            # print('Mapping trips to graph nodes...')
            poisson_rates_df = map_trip_to_graph_node(graph, poisson_rates_df, filtered_stations)

            # Save the Poisson rates to a CSV file
            # print("Saving the Poisson rates to a CSV file...")
            mon_str = str(params['month'][0]).zfill(2) + '-' + str(params['month'][-1]).zfill(2)
            rates_path = params['data_path'] + 'rates/' + mon_str + '/' + str(timeslot).zfill(2) + '/'
            if not os.path.exists(rates_path):
                os.makedirs(rates_path)
                print(f"Directory '{rates_path}' created.")
            poisson_rates_df.to_csv(rates_path + day.lower() + '-poisson-rates.csv', index=False)

            # Initialize the rate matrix
            # print("Initializing the rate matrix...")
            rate_matrix = initialize_rate_matrix(graph, poisson_rates_df)

            # Save the rate matrix to a CSV file
            # print("Saving the rate matrix to a CSV file...")
            matrix_path = params['data_path'] + 'matrices/' + mon_str + '/' + str(timeslot).zfill(2) + '/'
            if not os.path.exists(matrix_path):
                os.makedirs(matrix_path)
                print(f"Directory '{matrix_path}' created.")
            rate_matrix.to_csv(matrix_path + day.lower() + '-rate-matrix.csv', index=True)

            # Compute the global rate
            global_rate = kahan_sum(rate_matrix.to_numpy().flatten())
            global_rates[(day.lower(), timeslot)] = global_rate

            tbar.update(1)

    with open(params['data_path'] + params['global_rates_path'], 'wb') as f:
        pickle.dump(global_rates, f)

    # ---------- INTERPOLATE DATA ----------
    print("Finished preprocessing... Interpolating trip data.")

    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    nodes_dict = {node_id: (row['y'], row['x']) for node_id, row in nodes_gdf.iterrows()}

    radius = 500
    nearby_nodes_dict = {node_id: nodes_within_radius(node_id, nodes_dict, radius) for node_id in
                         tqdm(nodes_dict, desc="Building Nearby Nodes", dynamic_ncols=True)}

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

    # ---------- PREPROCESS TRUCK GRID ----------
    print("Finished interpolating trip data... Preprocessing truck grid data.")

    # Divide the graph into cells
    print("Dividing the graph into cells... ")
    cell_dict = divide_graph_into_cells(graph, 300)

    # Assign nodes to cells
    print("Assigning nodes to cells... ")
    assign_nodes_to_cells(graph, cell_dict)

    cell_to_remove = []
    for cell in cell_dict.values():
        if len(cell.nodes) == 0:
            cell_to_remove.append(cell.id)

    for cell_id in cell_to_remove:
        cell_dict.pop(cell_id)

    for cell in cell_dict.values():
        cell.set_center_node(graph)
        cell.set_diagonal()

    set_adjacent_cells(cell_dict)

    # plot_graph_with_grid(graph, cell_dict, plot_number_cells=True)
    #
    # plot_graph_with_grid(graph, cell_dict, plot_center_nodes=True)

    # Save dictionary to a file
    with open(params['data_path'] + params['cell_data_path'], 'wb') as file:
        pickle.dump(cell_dict, file)

    # ---------- PREPROCESS DISTANCE MATRIX ----------
    print("Finished preprocessing truck grid data... Preprocessing distance matrix.")

    # Initialize the distance matrix
    print("Initializing the distance matrix... ")
    distance_matrix = initialize_distance_matrix(graph)

    # Save the distance matrix to a CSV file
    print("Saving the distance matrix to a CSV file... ")
    distance_matrix.to_csv(params['data_path'] + params['distance_matrix_path'], index=True)

    # ---------- PREPROCESS NODES DICTIONARY ----------
    print("Finished preprocessing distance matrix... Preprocessing nodes dictionary.")

    # Compute the nodes dictionary
    user_radius = 250
    print("Creating nearby nodes dictionary...")
    user_nearby_nodes_dict = {node_id: nodes_within_radius(node_id, nodes_dict, user_radius) for node_id in
                         tqdm(nodes_dict, desc="Nodes")}

    # Save dictionary to a file
    with open(params['data_path'] + params['nearby_nodes_path'], 'wb') as file:
        pickle.dump(user_nearby_nodes_dict, file)


    print("Finished preprocessing all data.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess whole data.")
    parser.add_argument("--data_path", type=str, default="data/", help="The directory where the data will be saved.")

    args = parser.parse_args()
    if args.data_path:
        params['data_path'] = args.data_path
    main()