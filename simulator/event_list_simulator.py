import os
import random
import logging

import osmnx as ox
import networkx as nx
import pandas as pd

from tqdm import tqdm

from station import Station
from bike import Bike
from trip import Trip
from utils import generate_poisson_events, truncated_gaussian_speed, plot_graph

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
schedule_buffer = []
failures = 0
failures_from_path = 0
distance_matrix = pd.DataFrame()


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
        bikes = initialize_bikes(station, random.randint(0, 5))
        global system_bikes
        system_bikes.update(bikes)
        station.set_bikes(bikes)
        stations[index] = station

    return stations


def simulate_requests(time_interval: int, rate_matrix: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Simulate bike requests between stations based on the rate matrix.

    Parameters:
        - time_interval (int): The time interval for the simulation.

    Returns:
        - pd.DataFrame: A DataFrame containing the simulated request times for each station pair.
    """
    event_buffer = [(0, 0)] * time_interval
    indices = rate_matrix.index.tolist()

    tbar = tqdm(total=len(indices) ** 2, desc="Simulating Requests")

    for i in indices:
        for j in indices:
            rate = rate_matrix.at[i, j]
            if rate != 0:
                event_times = generate_poisson_events(rate, time_interval)
                for event_time in event_times:
                    event_buffer[event_time] = (i, j)

            tbar.update(1)

    return event_buffer


def compute_bike_travel_time(start_node: int, end_node: int, velocity_kmh: int = 15) -> int:
    """
    Compute the travel time between two nodes in the graph.

    Parameters:
        - G (nx.MultiDiGraph): The graph representing the road network.
        - start_node (int): The starting node of the trip.
        - end_node (int): The ending node of the trip.
        - velocity_kmh (int): The velocity of the bike in km/h. Default is 15 km/h.

    Returns:
        - int: The travel time in seconds.
    """
    velocity_mps = velocity_kmh / 3.6
    global distance_matrix
    trip_distance_meters = distance_matrix.at[start_node, end_node]
    travel_time_seconds = int(trip_distance_meters / velocity_mps)

    return travel_time_seconds


def schedule_request(start_time: int, end_time: int, start_location: Station, end_location: Station) -> bool:
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
        bike.set_availability(False)
        trip = Trip(start_time, end_time, start_location, end_location, bike)
        global schedule_buffer
        schedule_buffer.append(trip)
        logging.info("Trip scheduled: %s", trip)
        return True

    global failures
    failures += 1
    return False


def event_scheduler(time: int, station_dict: dict, request: tuple[int, int]):
    """
    Schedule events based on the request simulations.

    Parameters:
        - time (int): The current time in seconds.
        - stations (list): A list of Station objects.
        - request_simulations (pd.DataFrame): A DataFrame containing the simulated request times for each station pair.
    """
    if request != (0,0):
        # Schedule a trip if a request occurs
        trip_duration = compute_bike_travel_time(request[0], request[1], velocity_kmh=truncated_gaussian_speed())
        schedule_request(time, time + trip_duration, station_dict.get(request[0]), station_dict.get(request[1]))

    trips_to_remove = []

    # Drop finished trips
    global schedule_buffer
    for trip in schedule_buffer:
        if trip.get_end_time() < time:
            bike = trip.get_bike()
            bike.set_availability(True)
            end_location = trip.get_end_location()
            end_location.lock_bike(bike)
            trips_to_remove.append(trip)

    # Remove finished trips from the schedule buffer
    for trip in trips_to_remove:
        schedule_buffer.remove(trip)


def simulate_environment(time_interval: int, station_dict: dict, month: [int], day: str, time_slot: int):
    """
    Simulate the environment for a given time interval.

    Parameters:
        - time_interval (int): The time interval for the simulation.
        - stations (list): A list of Station objects.
        - request_simulations (pd.DataFrame): A DataFrame containing the simulated request times for each station pair.
    """
    # Initialize the rate matrix
    print("Initializing the rate matrix...")
    mon_str = str(month[0]).zfill(2) + '-' + str(month[-1]).zfill(2)
    matrix_path = params['data_path'] + 'matrices/' + mon_str + '/' + str(time_slot).zfill(2) + '/'
    # rate_matrix = pd.read_csv(matrix_path + day.lower() + '-rate-matrix.csv', index_col=0)
    rate_matrix = pd.read_csv('rescaled-interpolated-rate-matrix.csv', index_col='osmid')

    # Convert index and columns to integers
    rate_matrix.index = rate_matrix.index.astype(int)
    rate_matrix.columns = rate_matrix.columns.astype(int)

    # Simulate requests
    print("Simulating requests... ")
    request_buffer = simulate_requests(params['time_interval'], rate_matrix)

    for t in tqdm(range(0, time_interval), desc="Simulating Environment"):
        event_scheduler(t, station_dict, request_buffer[t])


def main():
    ox.settings.use_cache = True
    logging.basicConfig(filename='trip_output.log', level=logging.INFO, filemode='w')

    # Initialize distance matrix
    global distance_matrix
    distance_matrix = pd.read_csv(params['data_path'] + '/distance-matrix.csv', index_col=0)
    distance_matrix.index = distance_matrix.index.astype(int)
    distance_matrix.columns = distance_matrix.columns.astype(int)

    # Initialize the graph
    print("Initializing the graph... ")
    graph = initialize_graph(params['data_path'] + params['graph_file'])

    # plot_graph(graph)

    # Initialize stations
    print("Initializing stations... ")
    stations = initialize_stations(graph)

    # Simulate the environment
    print("Simulating the environment... ")
    simulate_environment(params['time_interval'], stations, params['month'], params['day'], params['time_slot'])

    print("Total number of failures: ", failures)
    logging.info("Total number of failures: %s", failures)

    print("Simulation completed.")


if __name__ == '__main__':
    main()
