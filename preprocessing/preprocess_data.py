import os
import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
import warnings

from geopy.distance import geodesic
from tqdm import tqdm
from simulator.utils import count_specific_day, connect_disconnected_neighbors

params = {
    'place': ["Cambridge, Massachusetts, USA"],
    'network_type': 'bike',

    'data_path': "data/",
    'graph_file': "cambridge_network.graphml",
    'year': 2022,
    'month': [9, 10],

    'day_of_week': ["Monday"],
}

# ----------------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------------

def compute_poisson_rates(df: pd.DataFrame, year: int, months: [int], day_of_week: str, time_slot: int) -> pd.DataFrame:
    """
    Compute the Poisson request rates for each station pair on a specific day and time slot.

    Parameters:
        - df (DataFrame): DataFrame containing the trip data.
        - day_of_week (str): Day of the week to filter by (e.g., "Monday", "Tuesday").
        - time_slot (int): Time slot (1-based), each representing a 3-hour interval starting from 1:00 am.

    Returns:
        - DataFrame: DataFrame containing the Poisson request rates for each station pair,
                     restricted to the specified day and time slot.
    """

    # Convert start time to datetime and extract the day of the week and hour
    df['starttime'] = pd.to_datetime(df['starttime'])
    df['day_of_week'] = df['starttime'].dt.day_name()
    df['hour'] = df['starttime'].dt.hour

    # Filter the data by the specified day of the week
    df_filtered = df[df['day_of_week'] == day_of_week]

    # Define the start and end of the time slot
    start_hour = 1 + time_slot * 3
    end_hour = start_hour + 3

    # Filter by the specified time slot
    df_filtered = df_filtered[(df_filtered['hour'] >= start_hour) & (df_filtered['hour'] < end_hour)]

    # Calculate the number of occurrences of the specified day in the month
    num_days = 0
    for month in months:
        num_days += count_specific_day(year, month, day_of_week)

    # Calculate the duration in seconds for the time slot across all days in the month
    total_time_seconds = num_days * 3 * 3600  # 3 hours per time slot

    # Group by start and end station details
    grouped_df = (df_filtered.groupby(['start station id', 'start station name', 'start station latitude',
                                       'start station longitude', 'end station id', 'end station name',
                                       'end station latitude', 'end station longitude'])
                  .size().reset_index(name='trip_count'))

    # Initialize the DataFrame to store Poisson rates
    rate_df = grouped_df.copy()

    # Compute the Poisson rate (lambda) for each station pair
    rate_df['lambda'] = rate_df['trip_count'] / total_time_seconds

    # Add day of the week and time slot information
    rate_df['day_of_week'] = day_of_week
    rate_df['time_slot'] = time_slot

    return rate_df


def maximum_distance_between_points(G: nx.MultiDiGraph) -> int:
    """
    Compute the maximum distance between any two nodes in the graph.

    Parameters:
        - G (networkx.MultiDiGraph): The graph representing the road network.

    Returns:
        - float: The maximum distance between any two nodes in the graph.
    """
    max_distance = 0
    for u, v, data in G.edges(data=True):
        # Get coordinates of the nodes
        u_coords = (G.nodes[u]['y'], G.nodes[u]['x'])
        v_coords = (G.nodes[v]['y'], G.nodes[v]['x'])

        # Calculate the distance between the nodes
        distance = geodesic(u_coords, v_coords).meters

        # Update maximum distance and edge if current distance is greater
        if distance > max_distance:
            max_distance = distance
    return max_distance


def is_within_graph_bounds(G: nx.MultiDiGraph, node_coords: tuple, nearest_node, threshold=500) -> bool:
    """
    Check if a point is within the bounds of the graph.

    Parameters:
        - G (networkx.MultiDiGraph): The graph representing the road network.
        - lat (float): The latitude of the point.
        - lon (float): The longitude of the point.
        - threshold (int): The maximum distance allowed between the point and the nearest node in the graph.

    Returns:
        - bool: True if the point is within the bounds of the graph, False otherwise.
    """
    # Find the nearest node to the point in the graph
    nearest_node_coords = (G.nodes[nearest_node]['y'], G.nodes[nearest_node]['x'])

    # Compute the distance between the point and the nearest node
    distance_to_nearest_node = geodesic((node_coords[0], node_coords[1]), nearest_node_coords).meters

    # Check if this distance is within the acceptable threshold
    return distance_to_nearest_node <= threshold


def map_trip_to_graph_node(G: nx.MultiDiGraph, trip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the start and end stations of the trips to the nearest nodes in the graph.

    Parameters:
        - G (networkx.MultiDiGraph): The graph representing the road network.
        - trip (DataFrame): The DataFrame containing the trip data.

    Returns:
        - DataFrame: The DataFrame with the start and end stations mapped to the nearest nodes.
    """
    dist_threshold = 100

    tbar = tqdm(total=trip_df.shape[0], desc="Mapping Trips to Graph Nodes", leave=False, position=0)

    for index, row in trip_df.iterrows():
        # Find the nearest node for the start and end stations
        start_node = ox.distance.nearest_nodes(G, Y=row['start station latitude'], X=row['start station longitude'])
        end_node = ox.distance.nearest_nodes(G, Y=row['end station latitude'], X=row['end station longitude'])

        # Check if the start and end stations are within the bounds of the graph
        if is_within_graph_bounds(G, (row['start station latitude'], row['start station longitude']), start_node, threshold=dist_threshold):
            if is_within_graph_bounds(G, (row['end station latitude'], row['end station longitude']), end_node, threshold=dist_threshold):
                # Update the start and end station IDs
                trip_df.at[index, 'start station id'] = start_node
                trip_df.at[index, 'end station id'] = end_node

                # Update the start and end station coordinates
                trip_df.at[index, 'start station latitude'] = G.nodes[start_node]['y']
                trip_df.at[index, 'start station longitude'] = G.nodes[start_node]['x']
                trip_df.at[index, 'end station latitude'] = G.nodes[end_node]['y']
                trip_df.at[index, 'end station longitude'] = G.nodes[end_node]['x']

                # Remove the start and end station names
                # trip_df.drop(columns=['start station name', 'end station name'], inplace=True)
            else:
                # Update the start station ID (end station ID becomes -1 if not within bounds)
                trip_df.at[index, 'start station id'] = start_node
                trip_df.at[index, 'end station id'] = 10000

                trip_df.at[index, 'start station latitude'] = G.nodes[start_node]['y']
                trip_df.at[index, 'start station longitude'] = G.nodes[start_node]['x']
                trip_df.at[index, 'end station latitude'] = 0.0
                trip_df.at[index, 'end station longitude'] = 0.0

        elif is_within_graph_bounds(G, (row['end station latitude'], row['end station longitude']), end_node, threshold=dist_threshold):
            # Update the end station ID (start station ID becomes -1 if not within bounds)
            trip_df.at[index, 'start station id'] = 10000
            trip_df.at[index, 'end station id'] = end_node

            trip_df.at[index, 'start station latitude'] = 0.0
            trip_df.at[index, 'start station longitude'] = 0.0
            trip_df.at[index, 'end station latitude'] = G.nodes[end_node]['y']
            trip_df.at[index, 'end station longitude'] = G.nodes[end_node]['x']
        else:
            trip_df.drop(index, inplace=True)

        tbar.update(1)


    # Reset index to handle any dropped rows properly
    trip_df.reset_index(drop=True, inplace=True)

    return trip_df

# ----------------------------------------------------------------------------------------------------------------------

def initialize_rate_matrix(G: nx.MultiDiGraph, rate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize a rate matrix based on the data rates.

    Parameters:
        - gdf_nodes (pd.DataFrame): The GeoDataFrame containing the nodes of the graph.
        - data_rates (pd.DataFrame): The DataFrame containing the data rates for bike trips.

    Returns:
        - pd.DataFrame: The rate matrix for the bike trips.
    """
    # Get node IDs and create an empty square DataFrame
    node_ids = ox.graph_to_gdfs(G, edges=False).index
    df = pd.DataFrame(index=node_ids, columns=node_ids, dtype='float64')

    # Add column and row for the station ID -1
    df.loc[10000] = 0.0
    df[10000] = 0.0

    # Initialize values to zero
    df = df.fillna(0.0)

    # Fill the rate matrix with the data rates
    for data_rates_index, data_rates_row in tqdm(rate_df.iterrows(), total=rate_df.shape[0], desc="Processing Rates"):
        i = data_rates_row['start station id']
        j = data_rates_row['end station id']
        rate = data_rates_row['lambda']
        df.at[i, j] = rate

    return df

# ----------------------------------------------------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in square")

    # Initialize the graph
    print("Initializing the graph... ")
    graph = initialize_graph(params['place'], params['network_type'], params['data_path'] + params['graph_file'],
                             remove_isolated_nodes=True, simplify_network=True)

    print(f'\nProcessing data for year {params["year"]} and month {params["month"]}...')
    trip_df = pd.DataFrame()
    for month in params['month']:
        path = params['data_path'] + 'trips/' + str(params['year']) + str(month).zfill(2) + '-bluebikes-tripdata.csv'
        if os.path.isfile(path):
            trip_df = pd.concat([trip_df, pd.read_csv(path)], ignore_index=True)
        else:
            print(f"Trip data file for month {month} does not exist. Skipping...")

    # tbar = tqdm(total=len(params['day_of_week']) * 8, desc="Processing Data")

    for day in params['day_of_week']:
        for timeslot in range(2, 3):
            print(f"\nProcessing data for {day} - Time Slot {timeslot}...")
            # Compute the rates for each station pair
            print("Computing Poisson rates... ")
            poisson_rates_df = compute_poisson_rates(trip_df, params['year'], params['month'], day, timeslot)

            # Transform the trip data to match the graph
            print("Mapping trips to graph nodes... ")
            poisson_rates_df = map_trip_to_graph_node(graph, poisson_rates_df)

            # Save the Poisson rates to a CSV file
            print("Saving the Poisson rates to a CSV file... ")
            mon_str = str(params['month'][0]).zfill(2) + '-' + str(params['month'][-1]).zfill(2)
            rates_path = params['data_path'] + 'rates/' + mon_str + '/' + str(timeslot).zfill(2) + '/'
            if not os.path.exists(rates_path):
                os.makedirs(rates_path)
                print(f"Directory '{rates_path}' created.")
            poisson_rates_df.to_csv(rates_path + day.lower() + '-poisson-rates.csv', index=False)

            # Initialize the rate matrix
            print("Initializing the rate matrix... ")
            rate_matrix = initialize_rate_matrix(graph, poisson_rates_df)

            # Save the rate matrix to a CSV file
            print("Saving the rate matrix to a CSV file... ")
            matrix_path = params['data_path'] + 'matrices/' + mon_str + '/' + str(timeslot).zfill(2) + '/'
            if not os.path.exists(matrix_path):
                os.makedirs(matrix_path)
                print(f"Directory '{matrix_path}' created.")
            rate_matrix.to_csv(matrix_path + day.lower() + '-rate-matrix.csv', index=True)

            # tbar.update(1)


if __name__ == '__main__':
    main()
