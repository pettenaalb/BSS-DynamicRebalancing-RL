import os
import pandas as pd
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic
from tqdm import tqdm

params = {
    'place': ["Cambridge, Massachusetts, USA",
            "Somerville, Massachusetts, USA"],
    'network_type': 'drive',

    'data_path': "data/",
    'graph_file': "cambridge_somerville_network.graphml",
    'year': 2022,
    'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

    'time_duration': 86400*31   # 1 month
}

def is_within_graph_bounds(graph, lat, lon, threshold=500):  # threshold in meters
    # Find the nearest node to the point in the graph
    nearest_node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)

    # Get the coordinates of this nearest node
    nearest_node_coords = (graph.nodes[nearest_node]['y'], graph.nodes[nearest_node]['x'])

    # Calculate the distance between the point and the nearest node
    distance_to_nearest_node = geodesic((lat, lon), nearest_node_coords).meters

    # Check if this distance is within the acceptable threshold
    return distance_to_nearest_node <= threshold


def transform_trips(G, trip_df: pd.DataFrame) -> pd.DataFrame:
    len_df = trip_df.shape[0]
    for index, row in tqdm(trip_df.iterrows(), total=len_df, desc="Processing Trips"):
        # Check if the start and end stations are within the bounds of the graph
        if (is_within_graph_bounds(G, row['start station latitude'], row['start station longitude'], threshold=50)
                and is_within_graph_bounds(G, row['end station latitude'], row['end station longitude'], threshold=50)):
            # Find the nearest node for the start and end stations
            start_node = ox.distance.nearest_nodes(G, Y=row['start station latitude'], X=row['start station longitude'])
            end_node = ox.distance.nearest_nodes(G, Y=row['end station latitude'], X=row['end station longitude'])

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

    # Reset index to handle any dropped rows properly
    trip_df.reset_index(drop=True, inplace=True)

    return trip_df

def compute_poisson_request_rates(df: pd.DataFrame, total_time_seconds=86400) -> pd.DataFrame:
    """
    Compute the Poisson request rates for each day and station pair.

    Parameters:
    df (DataFrame): DataFrame containing the trip data.
    total_time (int): Total time in seconds.
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

# Example usage
def main():
    if os.path.isfile(params['data_path'] + params['graph_file']):
        print("Network file already exists. Loading the network data... ")
        graph = ox.load_graphml(params['data_path'] + params['graph_file'])
        print("Network data loaded successfully.")
    else:
        print("Network file does not exist. Downloading the network data... ")
        graph = ox.graph_from_place(params['place'][0], network_type=params['network_type'])
        if len(params['place']) > 1:
            for place in params['place']:
                grp = ox.graph_from_place(place, network_type=params['network_type'])
                graph = nx.compose(graph, grp)
        ox.save_graphml(graph, params['data_path'] + params['graph_file'])
        print("Network data downloaded and saved successfully.")

    # Simplify the graph by consolidating intersections
    # G_proj = ox.project_graph(graph)
    # graph = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=False)

    for month in params['month']:
        print('\nProcessing data for ' + str(params['year']) + '-' + str(month).zfill(2) + '...')
        # Load the trip data
        print("Loading the trip data... ")
        trip_df = pd.read_csv(params['data_path'] + str(params['year']) + str(month).zfill(2) + '-bluebikes-tripdata.csv')

        # Compute the rates for each station pair
        print("Computing the Poisson request rates... ")
        poisson_rates_df = compute_poisson_request_rates(trip_df, total_time_seconds=params['time_duration'])

        # Transform the trip data to match the graph
        print("Transforming the trip data... ")
        poisson_rates_df = transform_trips(graph, poisson_rates_df)

        # Save the Poisson rates to a CSV file
        print("Saving the Poisson rates to a CSV file... ")
        poisson_rates_df.to_csv(params['data_path'] + str(params['year']) + str(month).zfill(2) + '-poisson-rates.csv', index=False)

        print("Poisson rates saved successfully.")
        #print(poisson_rates_df)

if __name__ == '__main__':
    main()
