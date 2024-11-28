import os
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx

from scipy.stats import truncnorm
from geopy.distance import distance
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gymnasium_env.simulator.cell import Cell
    from gymnasium_env.simulator.station import Station
    from gymnasium_env.simulator.bike import Bike

# ----------------------------------------------------------------------------------------------------------------------

def kahan_sum(arr):
    total = 0.0
    c = 0.0  # A running compensation for lost low-order bits.
    for value in arr:
        y = value - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


def compute_distance(coords1, coords2):
    """
    Calculate the distance between two pairs of coordinates in meters.

    Parameters:
        - coords1: A tuple (lat1, lon1) for the first coordinate.
        - coords2: A tuple (lat2, lon2) for the second coordinate.

    Returns:
        - distance_in_meters: The distance in meters between the two coordinates.
    """
    # Calculate the geodesic distance
    distance_in_meters = distance(coords1, coords2).meters
    return distance_in_meters


def generate_poisson_events(rate, time_duration) -> list[int]:
    """
    Generate Poisson events within a specified time duration.

    Parameters:
        - rate (float): The average rate of events per unit time.
        - time_duration (float): The total time duration in which events can occur.

    Returns:
        - list: A list of event times occurring within the specified time duration.
    """
    # uniform distribution of arrival times
    num_events = np.random.poisson(rate * time_duration)
    event_times = np.sort(np.random.uniform(0, time_duration, num_events)).astype(int).tolist()

    return event_times


def convert_seconds_to_hours_minutes(seconds) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}"


def truncated_gaussian(lower=5, upper=25, mean=15, std_dev=5):
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    truncated_normal = truncnorm(a, b, loc=mean, scale=std_dev)
    speed = truncated_normal.rvs()
    return speed

# ----------------------------------------------------------------------------------------------------------------------

def load_cells_from_csv(filename) -> dict[int, "Cell"]:
    from gymnasium_env.simulator.cell import Cell
    df = pd.read_csv(filename)
    cells = {row['id']: Cell.from_dict(row) for _, row in df.iterrows()}
    return cells

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


def initialize_bikes(stn: "Station", n: int = 0) -> dict[int, "Bike"]:
    """
    Initialize a list of bikes at a station.

    Parameters:
        - stn (Station): The station where the bikes are located.
        - n (int): The number of bikes to initialize. Default is 0.

    Returns:
        - dict: A dictionary containing the bikes at the station.
    """
    from gymnasium_env.simulator.bike import Bike
    bikes = {}
    for i in range(n):
        bike = Bike(stn=stn)
        bikes[bike.get_bike_id()] = bike
    return bikes


def initialize_stations(G: nx.MultiDiGraph, bikes_per_station: dict[int, int] = None, pmf_matrix: pd.DataFrame = None,
                        global_rate: float = None) -> tuple[dict[int, "Station"], dict[int, "Bike"]]:
    """
    Initialize a list of stations based on the nodes of the graph.

    Parameters:
        - G (nx.MultiDiGraph): The graph representing the road network.

    Returns:
        - dict: A dictionary containing the stations in the network.
    """
    from gymnasium_env.simulator.station import Station
    gdf_nodes = ox.graph_to_gdfs(G, edges=False)

    stations = {}
    sys_bikes = {}

    for index, row in gdf_nodes.iterrows():
        if pmf_matrix is not None and global_rate is not None:
            request_rate =  kahan_sum(pmf_matrix.loc[index].values) * global_rate
        else:
            request_rate = 0.0
        station = Station(index, row["y"], row["x"], request_rate=request_rate)
        if bikes_per_station is not None:
            bikes = initialize_bikes(station, bikes_per_station.get(index))
        else:
            bikes = initialize_bikes(station, np.random.randint(0, 2))
        sys_bikes.update(bikes)
        station.set_bikes(bikes)
        stations[index] = station

    stations[10000] = Station(10000, 0, 0)

    return stations, sys_bikes
