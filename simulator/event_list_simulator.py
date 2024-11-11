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
distance_matrix = pd.DataFrame()
total_trips = 0

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
    bikes = {f"bike_{i}": Bike(stn=stn) for i in range(n)}
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


def set_trip(bike: Bike, start_time: int, distance: int, start_location: Station, end_location: Station, schedule_buffer: list):
    bike.set_battery(bike.get_battery() - distance/1000)

    velocity_kmh = truncated_gaussian(5, 25, 15, 5)
    velocity_mps = velocity_kmh / 3.6
    travel_time_seconds = int(distance / velocity_mps)

    trip = Trip(start_time, start_time + travel_time_seconds, start_location, end_location, bike)
    schedule_buffer.append(trip)
    global total_trips
    total_trips += 1
    logging.info("Trip scheduled: %s", trip)



def schedule_request(start_time: int, distance: int, start_location: Station, end_location: Station,
                     schedule_buffer: list, nearby_nodes_dict: dict[str, tuple], station_dict: dict) -> int:
    """
    Schedule a trip between two stations.

    Parameters:
        - start_time (int): The start time of the trip.
        - end_time (int): The end time of the trip.
        - start_location (Station): The starting location of the trip.
        - end_location (Station): The ending location of the trip.

    Returns:
        - bool: True if the trip was scheduled successfully, False otherwise.
    """


    if len(start_location.get_bikes()) > 0:
        bike = start_location.unlock_bike()
        if bike.get_battery() > distance/1000:
            set_trip(bike, start_time, distance, start_location, end_location, schedule_buffer)
            return 0
        else:
            start_location.lock_bike(bike)
            logging.warning(f"No charged bikes available from station {start_location.get_station_id()} to station {end_location.get_station_id()}")
            return 1
    else:
        # List of reordered nearby nodes based on distance matrix
        nodes_dist_dict = {node_id: distance_matrix.at[start_location.get_station_id(), node_id] for node_id in nearby_nodes_dict}
        sorted_nodes = {k: v for k, v in sorted(nodes_dist_dict.items(), key=lambda item: item[1])}

        for node_id in sorted_nodes:
            if len(station_dict[node_id].get_bikes()) > 0:
                bike = station_dict[node_id].unlock_bike()
                if bike.get_battery() > distance/1000:
                    set_trip(bike, start_time, distance, start_location, end_location, schedule_buffer)
                    return 0

        logging.warning(f"No bike available from station {start_location.get_station_id()} to station {end_location.get_station_id()}")
        return 1


def schedule_in_out_system_request(start_time: int, start_location: Station, end_location: Station, leaving: bool,
                                   schedule_buffer: list, nearby_nodes_dict: dict[str, tuple], station_dict: dict) -> int:
    global system_bikes, total_trips
    if leaving:
        distance = truncated_gaussian(1, 7, 3, 1).to(float)
        return schedule_request(start_time, distance, start_location, end_location, schedule_buffer, nearby_nodes_dict, station_dict)
    else:
        if len(start_location.get_bikes()) > 0:
            bike = start_location.unlock_bike()
            end_location.lock_bike(bike)
        else:
            bike = Bike(stn=end_location)
            battery = truncated_gaussian(0, 50, 25, 5).to(float)
            bike.set_battery(battery)
            end_location.lock_bike(bike)
            system_bikes[bike.get_bike_id()] = bike

        trip = Trip(start_time, start_time, start_location, end_location, bike)
        total_trips += 1
        logging.info("Trip scheduled: %s", trip)

        return 0


def event_scheduler(time: int, station_dict: dict, request: int, pmf: pd.DataFrame, schedule_buffer: list,
                    nearby_nodes_dict: dict[str, dict[str, tuple]]) -> int:
    """
    Schedule events based on the request simulations.

    Parameters:
        - time (int): The current time in seconds.
        - stations (list): A list of Station objects.
        - request_simulations (pd.DataFrame): A DataFrame containing the simulated request times for each station pair.
    """
    global system_bikes, outside_system_bikes

    failure = 0
    if request != 0:
        stn_pair = tuple(np.random.choice(pmf['id'], p=pmf['value']))
        if stn_pair[0] == 10000 and stn_pair[1] == 10000:
            raise ValueError("Request simulation not implemented (id 100000).")
        if stn_pair[0] == 10000:
            failure = schedule_in_out_system_request(time, station_dict.get(stn_pair[0]), station_dict.get(stn_pair[1]),
                                                     False, schedule_buffer, nearby_nodes_dict[stn_pair[0]], station_dict)
        elif stn_pair[1] == 10000:
            failure = schedule_in_out_system_request(time, station_dict.get(stn_pair[0]), station_dict.get(stn_pair[1]),
                                                     True, schedule_buffer, nearby_nodes_dict[stn_pair[0]], station_dict)
        else:
            global distance_matrix
            distance = distance_matrix.at[stn_pair[0], stn_pair[1]]
            failure = schedule_request(time, distance, station_dict.get(stn_pair[0]), station_dict.get(stn_pair[1]),
                                       schedule_buffer, nearby_nodes_dict[stn_pair[0]], station_dict)

    # Drop finished trips
    trips_to_remove = []
    for trip in schedule_buffer:
        if trip.get_end_time() < time:
            bike = trip.get_bike()
            end_location = trip.get_end_location()
            end_location.lock_bike(bike)
            if end_location.get_station_id == 10000:
                system_bikes.pop(bike.get_station_id())
                outside_system_bikes[bike.get_station_id()] = bike
            trips_to_remove.append(trip)

    # Remove finished trips from the schedule buffer
    for trip in trips_to_remove:
        schedule_buffer.remove(trip)

    return failure

# ----------------------------------------------------------------------------------------------------------------------

def simulate_environment(time_interval: int, station_dict: dict, pmf: pd.DataFrame, rate: float, timeslot: int,
                         nearby_nodes_dict: dict[str, dict[str, tuple]]) -> int:
    """
    Simulate the environment for a given time interval.

    Parameters:
        - time_interval (int): The time interval for the simulation.
        - stations (list): A list of Station objects.
        - request_simulations (pd.DataFrame): A DataFrame containing the simulated request times for each station pair.
    """
    # Simulate requests
    event_times = generate_poisson_events(rate, time_interval)
    event_buffer = [0] * time_interval
    for event_time in event_times:
        event_buffer[event_time] = 1

    schedule_buffer = []
    failures = 0

    for t in tqdm(range(0, time_interval), desc="Simulating Environment"):
        failures += event_scheduler(t + (timeslot*3 + 1)*3600, station_dict, event_buffer[t], pmf, schedule_buffer, nearby_nodes_dict)

    return failures

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
    global distance_matrix
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
    global_rate = kahan_sum(rate_matrix.to_numpy().flatten())

    # Simulate the environment
    print("Simulating the environment... ")
    failures = simulate_environment(params['time_interval'], stations, flattened_pmf, global_rate, params['time_slot'], nearby_nodes_dict)

    print(f"\nTotal number of trips: {total_trips}")
    logging.info(f"Total number of trips: {total_trips}")

    print(f"Total number of failures: {failures}")
    logging.info(f"Total number of failures: {failures}")

    print("Simulation completed.")


if __name__ == '__main__':
    main()
