import os
import pandas as pd
import osmnx as ox
import networkx as nx

from geopy.distance import geodesic
from tqdm import tqdm

from simulator.utils import plot_graph

params = {
    'place': ["Cambridge, Massachusetts, USA"],
    'network_type': 'bike',

    'data_path': "data/",
    'graph_file': "cambridge_network.graphml",
    'year': 2022,
    'month': [1],

    'time_duration': [31*24*3600]
}


def initialize_graph(places: [str], network_type: str, graph_path: str = None, simplify_network: bool = False,
                     remove_isolated_nodes: bool = False) -> nx.MultiDiGraph:
    if os.path.isfile(graph_path):
        print("Network file already exists. Loading the network data... ")
        graph = ox.load_graphml(graph_path)
        print("Network data loaded successfully.")
    else:
        print("Network file does not exist. Downloading the network data... ")
        graph = ox.graph_from_place(places[0], network_type=network_type)

        if len(places) > 1:
            for place in places:
                grp = ox.graph_from_place(place, network_type=network_type)
                graph = nx.compose(graph, grp)

        # OSM data are sometime incomplete so we use the speed module of osmnx to add missing edge speeds and travel times
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)

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
    len_df = trip_df.shape[0]
    dist_threshold = maximum_distance_between_points(G) + 5
    for index, row in tqdm(trip_df.iterrows(), total=len_df, desc="Processing Trips"):
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
                trip_df.drop(index, inplace=True)
        else:
            trip_df.drop(index, inplace=True)

    # Reset index to handle any dropped rows properly
    trip_df.reset_index(drop=True, inplace=True)

    return trip_df


def compute_poisson_request_rates(df: pd.DataFrame, total_time_seconds=86400) -> pd.DataFrame:
    """
    Compute the Poisson request rates for each day and station pair.

    Parameters:
        - df (DataFrame): DataFrame containing the trip data.
        - total_time (int): Total time in seconds.

    Returns:
        - DataFrame: DataFrame containing the Poisson request rates for each day and station pair.
    """

    # Group by start station, and end station
    grouped_df = (df.copy().groupby(['start station id', 'start station name', 'start station latitude',
                                     'start station longitude', 'end station id', 'end station name',
                                     'end station latitude', 'end station longitude'])
                  .size().reset_index(name='trip_count'))

    # Initialize the DataFrame to store Poisson rates
    rate_df = grouped_df.copy()

    # Compute the Poisson rate (lambda) for each day and station pair
    rate_df['lambda'] = rate_df['trip_count'] / total_time_seconds

    return rate_df


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
    node_ids = list(G.nodes)
    df = pd.DataFrame(index=node_ids, columns=node_ids, dtype=float)

    # Initialize values to zero
    df = df.fillna(0)

    for data_rates_index, data_rates_row in tqdm(rate_df.iterrows(), total=rate_df.shape[0], desc="Processing Rates"):
        i = data_rates_row['start station id']
        j = data_rates_row['end station id']
        rate = data_rates_row['lambda']
        df.at[i, j] = rate

    return df


# Example usage
def main():
    # Initialize the graph
    print("Initializing the graph... ")
    graph = initialize_graph(params['place'], params['network_type'], params['data_path'] + params['graph_file'],
                             remove_isolated_nodes=True, simplify_network=True)

    plot_graph(graph)

    for month, time_duration in zip(params['month'], params['time_duration']):
        print('\nProcessing data for ' + str(params['year']) + '-' + str(month).zfill(2) + '...')
        # Load the trip data
        print("Loading the trip data... ")
        trip_df = pd.read_csv(params['data_path'] + "trips/" + str(params['year']) + str(month).zfill(2) + '-bluebikes-tripdata.csv')

        # Compute the rates for each station pair
        print("Computing the Poisson request rates... ")
        poisson_rates_df = compute_poisson_request_rates(trip_df, total_time_seconds=time_duration)

        # Transform the trip data to match the graph
        print("Transforming the trip data... ")
        poisson_rates_df = map_trip_to_graph_node(graph, poisson_rates_df)

        # Save the Poisson rates to a CSV file
        print("Saving the Poisson rates to a CSV file... ")
        poisson_rates_df.to_csv(params['data_path'] + "rates/" + str(params['year']) + str(month).zfill(2) + '-poisson-rates.csv', index=False)

        # Load the Poisson rates, uncomment the above code to generate the rates
        # print("Loading the Poisson rates... ")
        # poisson_rates_df = pd.read_csv(params['data_path'] + "rates/" + str(params['year']) + str(month).zfill(2) + '-poisson-rates.csv')

        # Initialize the rate matrix
        print("Initializing the rate matrix... ")
        rate_matrix = initialize_rate_matrix(graph, poisson_rates_df)

        # Save the rate matrix to a CSV file
        print("Saving the rate matrix to a CSV file... ")
        rate_matrix.to_csv(params['data_path'] + "matrices/" + str(params['year']) + str(month).zfill(2) + '-rate-matrix.csv', index=True)

        print("Poisson rates saved successfully.")
        #print(poisson_rates_df)

if __name__ == '__main__':
    main()
