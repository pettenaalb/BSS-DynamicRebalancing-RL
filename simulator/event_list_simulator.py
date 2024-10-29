import os
import random

import osmnx as ox
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

from tqdm import tqdm
from utils import poisson_simulation
from station import Station
from bike import Bike
from trip import Trip

params = {
    'place': ["Cambridge, Massachusetts, USA"],
    'network_type': 'drive',

    'data_path': "../data/",
    'graph_file': "cambridge_network.graphml",
    'year': 2022,
    'month': 1,

    'time_interval': 3600   # 1 hour
}

system_bikes = []
schedule_buffer = []
failures = 0


def initialize_bikes(stn: Station, n: int = 0) -> list:
    """
    Initialize a list of bikes at a station.

    Parameters:
        - stn (Station): The station where the bikes are located.
        - n (int): The number of bikes to initialize. Default is 0.

    Returns:
        - list: A list of Bike objects at the station.
    """
    bikes = []
    for i in range(n):
        bikes.append(Bike(stn))
    return bikes


def initialize_stations(G: nx.MultiDiGraph) -> list:
    """
    Initialize a list of stations based on the nodes of the graph.

    Parameters:
        - G (nx.MultiDiGraph): The graph representing the road network.

    Returns:
        - list: A list of Station objects.
    """

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    stations = []

    for index, row in gdf_nodes.iterrows():
        stations.append(Station(index, row["y"], row["x"]))
        bikes = initialize_bikes(stations[-1], random.randint(5, 10))
        global system_bikes
        system_bikes.extend(bikes)
        stations[-1].set_bikes(bikes)

    return stations


def simulate_requests(time_interval: int, rate_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate bike requests between stations based on the rate matrix.

    Parameters:
        - time_interval (int): The time interval for the simulation.

    Returns:
        - pd.DataFrame: A DataFrame containing the simulated request times for each station pair.
    """
    indices = rate_matrix.index.tolist()
    request_simulations = pd.DataFrame([[[] for _ in range(len(indices))] for _ in range(len(indices))],
                                       index=indices, columns=indices)

    tbar = tqdm(total=len(indices) ** 2, desc="Simulating Requests")

    for i in indices:
        for j in indices:
            rate = rate_matrix.at[i, j]
            if rate != 0:
                _, event_times, _ = poisson_simulation(rate, time_interval)
                request_simulations.at[i, j] = event_times
            else:
                request_simulations.at[i, j] = None

            tbar.update(1)

    return request_simulations


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
        print("Trip scheduled: ", trip)
        return True

    global failures
    failures += 1
    return False


def event_scheduler(time: int, stations: list, request_simulations: pd.DataFrame):
    """
    Schedule events based on the request simulations.

    Parameters:
        - time (int): The current time in seconds.
        - stations (list): A list of Station objects.
        - request_simulations (pd.DataFrame): A DataFrame containing the simulated request times for each station pair.
    """
    tbar = tqdm(total=len(stations) ** 2, desc="Processing Events")

    # Check for requests
    for i, start_station in enumerate(stations):
        for j, end_station in enumerate(stations):
            event_times = request_simulations.at[start_station.get_station_id(), end_station.get_station_id()]
            if event_times is not None:
                if time in event_times:
                    # Schedule a trip if a request occurs, with a random trip duration between 5 and 15 minutes
                    trip_duration = random.randint(5, 15) * 60
                    schedule_request(time, time + trip_duration, start_station, end_station)

    # Drop finished trips
    global schedule_buffer
    for trip in schedule_buffer:
        if trip.get_end_time() <= time:
            bike = trip.get_bike()
            bike.set_availability(True)
            end_location = trip.get_end_location()
            end_location.lock_bike(bike)
            schedule_buffer.remove(trip)

def simulate_environment(time_interval: int, stations: list, request_simulations: pd.DataFrame):
    """
    Simulate the environment for a given time interval.

    Parameters:
        - time_interval (int): The time interval for the simulation.
        - stations (list): A list of Station objects.
        - request_simulations (pd.DataFrame): A DataFrame containing the simulated request times for each station pair.
    """
    for t in tqdm(range(0, time_interval), desc="Simulating Environment"):
        event_scheduler(t, stations, request_simulations)


def plot_poisson_process(data, index):
    """Plot the Poisson process for a given list."""
    plt.plot(data, marker='o', linestyle='-', label=f'Process {index}')
    plt.title(f'Poisson Process {index}')
    plt.xlabel('Time')
    plt.ylabel('Number of Events')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    ox.settings.use_cache = True

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

    # Initialize stations
    print("Initializing stations... ")
    stations = initialize_stations(graph)

    # Initialize the rate matrix
    print("Initializing the rate matrix...")
    rate_matrix = pd.read_csv(params['data_path'] + "matrices/" + str(params['year']) + str(params['month']).zfill(2) + '-rate-matrix.csv', index_col=0)

    # Convert index and columns to integers
    rate_matrix.index = rate_matrix.index.astype(int)
    rate_matrix.columns = rate_matrix.columns.astype(int)

    # Simulate requests
    print("Simulating requests... ")
    request_simulations = simulate_requests(params['time_interval'], rate_matrix)

    # rt = pd.read_csv(params['data_path'] + "rates/" + str(params['year']) + str(params['month']).zfill(2) + '-poisson-rates.csv')
    #
    # print(rate_matrix.at[rt['start station id'][0], rt['end station id'][0]])
    # req_list = request_simulations.at[rt['start station id'][0], rt['end station id'][0]]
    # print(req_list)

    # Simulate the environment
    print("Simulating the environment... ")
    simulate_environment(params['time_interval'], stations, request_simulations)


if __name__ == '__main__':
    main()
