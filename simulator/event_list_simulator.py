import os
import random
import logging

import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np

from tqdm import tqdm
from station import Station
from bike import Bike
from trip import Trip
from event import EventType, Event
from utils import generate_poisson_events, truncated_gaussian, kahan_sum, nodes_within_radius

# ----------------------------------------------------------------------------------------------------------------------

params = {
    'data_path': "../data/",
    'graph_file': "cambridge_network.graphml",
    'network_type': 'bike',

    'year': 2022,
    'month': [9, 10],
    'day': "monday",
    'time_slot': 2,
    'time_interval': 3600*3   # 3 hour
}

system_bikes = {}
outside_system_bikes = {}

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


def initialize_bikes(stn: Station, n: int = 0) -> dict:
    """
    Initialize a list of bikes at a station.

    Parameters:
        - stn (Station): The station where the bikes are located.
        - n (int): The number of bikes to initialize. Default is 0.

    Returns:
        - dict: A dictionary containing the bikes at the station.
    """
    bikes = {}
    for i in range(n):
        bike = Bike(stn=stn)
        bikes[bike.get_bike_id()] = bike
    return bikes


def initialize_stations(G: nx.MultiDiGraph) -> dict:
    """
    Initialize a list of stations based on the nodes of the graph.

    Parameters:
        - G (nx.MultiDiGraph): The graph representing the road network.

    Returns:
        - dict: A dictionary containing the stations in the network.
    """

    gdf_nodes = ox.graph_to_gdfs(G, edges=False)

    stations = {}

    for index, row in gdf_nodes.iterrows():
        station = Station(index, row["y"], row["x"])
        bikes = initialize_bikes(station, random.randint(0, 2))
        global system_bikes
        system_bikes.update(bikes)
        station.set_bikes(bikes)
        stations[index] = station

    stations[10000] = Station(10000, 0, 0)

    return stations

# ----------------------------------------------------------------------------------------------------------------------

def departure_handler(trip: Trip, station_dict: dict, nearby_nodes_dict: dict[str, dict[str, tuple]],
                      distance_matrix: pd.DataFrame) -> Trip:
    start_station = trip.get_start_location()
    start_station_id = start_station.get_station_id()

    if len(start_station.get_bikes()) > 0:
        bike = start_station.unlock_bike()
        if bike.get_battery() > trip.get_distance()/1000:
            trip.set_bike(bike)
            trip.set_failed(False)
            return trip
        else:
            start_station.lock_bike(bike)

    nodes_dist_dict = {node_id: distance_matrix.at[start_station_id, node_id] for node_id in nearby_nodes_dict[start_station_id]}
    sorted_nodes = {k: v for k, v in sorted(nodes_dist_dict.items(), key=lambda item: item[1])}
    for node_id in sorted_nodes:
        if len(station_dict[node_id].get_bikes()) > 0:
            bike = station_dict[node_id].unlock_bike()
            if bike.get_battery() > trip.get_distance()/1000:
                trip.set_bike(bike)
                trip.set_failed(False)
                return trip
            else:
                start_station.lock_bike(bike)

    trip.set_failed(True)
    return trip


def arrival_handler(trip: Trip):
    global system_bikes, outside_system_bikes
    if trip.is_failed():
        return

    start_station = trip.get_start_location()
    end_station = trip.get_end_location()
    bike = trip.get_bike()
    end_station.lock_bike(bike)

    if end_station.get_station_id() == 10000:
        outside_system_bikes[bike.get_bike_id()] = system_bikes.pop(bike.get_bike_id())

    if start_station.get_station_id() == 10000:
        system_bikes[bike.get_bike_id()] = outside_system_bikes.pop(bike.get_bike_id())

# ----------------------------------------------------------------------------------------------------------------------

def simulate_environment(time_interval: int, rate: float, pmf: pd.DataFrame, station_dict: dict,
                         nearby_nodes_dict: dict[str, dict[str, tuple]], distance_matrix: pd.DataFrame) -> tuple[int, int]:
    """
    Simulate the environment for a given time interval.

    Parameters:
        - time_interval (int): The time interval for the simulation.
        - stations (list): A list of Station objects.
        - request_simulations (pd.DataFrame): A DataFrame containing the simulated request times for each station pair.
    """
    # Simulate requests
    event_times = generate_poisson_events(rate, time_interval)
    event_buffer = []
    tbar = tqdm(total=len(event_times), desc="Setting events", position=0, leave=True)
    for event_time in event_times:
        random_value = np.random.rand()
        stn_pair = tuple(pmf.iloc[np.searchsorted(pmf['cumsum'].values, random_value)]['id'])
        if stn_pair[0] == 10000:
            ev_t = event_time + 3600*(params['time_slot'] + 1)
            trip = Trip(ev_t, ev_t, station_dict[stn_pair[0]], station_dict[stn_pair[1]])
            arr_event = Event(time=event_time, event_type=EventType.ARRIVAL, trip=trip)
            event_buffer.append(arr_event)
        else:
            velocity_kmh = truncated_gaussian(5, 25, 15, 5)
            if stn_pair[1] == 10000:
                distance = truncated_gaussian(2, 7, 4.5, 1)
            else:
                distance = distance_matrix.at[stn_pair[0], stn_pair[1]]
            travel_time_seconds = int(distance * 3.6 / velocity_kmh)
            ev_t = event_time + 3600*(3*params['time_slot'] + 1)
            trip = Trip(ev_t, ev_t + travel_time_seconds, station_dict[stn_pair[0]],
                        station_dict[stn_pair[1]], distance=distance)
            dep_event = Event(time=event_time, event_type=EventType.DEPARTURE, trip=trip)
            arr_event = Event(time=event_time + travel_time_seconds, event_type=EventType.ARRIVAL, trip=trip)
            event_buffer.append(dep_event)
            event_buffer.append(arr_event)

        tbar.update(1)

    # Sort the event buffer based on time
    event_buffer.sort(key=lambda x: x.time)
    failures = 0
    total_trips = 0

    tbar = tqdm(total=len(event_buffer), desc="Processing events", position=0, leave=True)

    for event in event_buffer:
        if event.is_departure():
            trip = departure_handler(event.get_trip(), station_dict, nearby_nodes_dict, distance_matrix)
            if trip.is_failed():
                failures += 1
                logging.warning(f"No bike available from station {trip.get_start_location().get_station_id()} to station {trip.get_end_location().get_station_id()}")
            else:
                total_trips += 1
                logging.info("Trip scheduled: %s", trip)
        else:
            arrival_handler(event.get_trip())

        tbar.update(1)

    return total_trips, failures

# ----------------------------------------------------------------------------------------------------------------------

def main():
    ox.settings.use_cache = True
    logging.basicConfig(filename='trip_output.log', level=logging.INFO, filemode='w')

    # Initialize the graph
    print("Initializing the graph... ")
    graph = initialize_graph(params['data_path'] + params['graph_file'])

    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    nodes_dict = {node_id: (row['y'], row['x']) for node_id, row in nodes_gdf.iterrows()}

    radius = 100
    nearby_nodes_dict = {node_id: nodes_within_radius(node_id, nodes_dict, radius) for node_id in nodes_dict}

    # Initialize distance matrix
    distance_matrix = pd.read_csv(params['data_path'] + '/distance-matrix.csv', index_col=0)
    distance_matrix.index = distance_matrix.index.astype(int)
    distance_matrix.columns = distance_matrix.columns.astype(int)

    # Initialize stations
    print("Initializing stations... ")
    stations = initialize_stations(graph)

    # Initialize the rate matrix
    mon_str = str(params['month'][0]).zfill(2) + '-' + str(params['month'][-1]).zfill(2)
    matrix_path = params['data_path'] + 'matrices/' + mon_str + '/' + str(params['time_slot']).zfill(2) + '/'
    pmf_matrix = pd.read_csv(matrix_path + params['day'].lower() + '-pmf-matrix.csv', index_col='osmid')
    rate_matrix = pd.read_csv(matrix_path + params['day'].lower() + '-rate-matrix.csv', index_col='osmid')

    # Convert index and columns to integers
    pmf_matrix.index = pmf_matrix.index.astype(int)
    pmf_matrix.columns = pmf_matrix.columns.astype(int)

    pmf_matrix.loc[10000, 10000] = 0.0

    # Convert into one vector
    values = pmf_matrix.values.flatten()
    ids = [(row, col) for row in pmf_matrix.index for col in pmf_matrix.columns]

    # Compute flattened pmf and global rate
    flattened_pmf = pd.DataFrame({'id': ids, 'value': values})
    flattened_pmf['cumsum'] = np.cumsum(flattened_pmf['value'].values)
    global_rate = kahan_sum(rate_matrix.to_numpy().flatten())

    # Simulate the environment
    print("Simulating the environment... ")
    total_trips, failures = simulate_environment(params['time_interval'], global_rate, flattened_pmf, stations, nearby_nodes_dict, distance_matrix)

    print(f"\nTotal number of trips: {total_trips}")
    logging.info(f"Total number of trips: {total_trips}")

    print(f"Total number of failures: {failures}")
    logging.info(f"Total number of failures: {failures}")

    print("Simulation completed.")


if __name__ == '__main__':
    main()
