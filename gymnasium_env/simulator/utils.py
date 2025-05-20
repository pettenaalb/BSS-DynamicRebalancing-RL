import os
import logging
import math
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import torch.nn as nn

from scipy.stats import truncnorm
from geopy.distance import distance
from typing import TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from gymnasium_env.simulator.cell import Cell
    from gymnasium_env.simulator.station import Station
    from gymnasium_env.simulator.bike import Bike
    from gymnasium_env.simulator.truck import Truck
    from gymnasium_env.simulator.trip import Trip

# ----------------------------------------------------------------------------------------------------------------------

class Actions(Enum):
    STAY = 0
    RIGHT = 1
    UP = 2
    LEFT = 3
    DOWN = 4
    DROP_BIKE = 5
    PICK_UP_BIKE = 6
    # TURN OFF THIS TO DISABLE BATTERY CHARGE
    CHARGE_BIKE = 7


# Action History Encoder with Embeddings
class ActionHistoryEncoder(nn.Module):
    def __init__(self, num_actions=7, embedding_dim=4, history_length=4):
        super().__init__()
        # Maps action index to an embedding
        self.embedding = nn.Embedding(num_actions, embedding_dim)
        self.history_length = history_length

    def forward(self, action_history):
        embedded_actions = self.embedding(action_history)
        return embedded_actions.view(action_history.shape[0], -1)


class Logger:
    def __init__(self, log_file: str, is_logging: bool = False):
        if is_logging:
            logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')
            self.logger = logging.getLogger('env_logger')
            self.is_logging = is_logging
        else:
            self.logger = None
            self.is_logging = False

    def new_log_line(self):
        if self.is_logging:
            self.logger.info("--------------------------------------------------------")

    def log_starting_action(self, action: str, t: int):
        if self.is_logging:
            self.logger.info(f'START ACTION: '
                             f'\n - {action}'
                             f'\n - Time: {t}'
                             f'\n - Steps needed: {int(math.ceil(t / 30))}')

    def log_ending_action(self, time: str):
        if self.is_logging:
            self.logger.info(f'Action completed successfully - Time: {time}')

    def log_state(self, step: int, time: str):
        if self.is_logging:
            self.logger.info(f'State S_{step} - Time: {time}')

    def log_truck(self, truck: "Truck"):
        if self.is_logging:
            self.logger.info(f"\nTRUCK:"
                             f"\n - CELL: {truck.cell.get_id()} - {truck.cell.get_center_node()}"
                             f"\n - POSITION: {truck.position}"
                             f"\n - LOAD: {truck.current_load} bikes")

    def log_no_available_bikes(self, start_station: int, end_station: int):
        if self.is_logging:
            self.logger.warning(f"No bike available from station {start_station} to station {end_station}")

    def log_trip(self, trip: "Trip"):
        if self.is_logging:
            self.logger.info("Trip scheduled %s", trip)

    def log(self, message: str):
        if self.is_logging:
            self.logger.info(f"\n{message}\n")

    def set_logging(self, is_logging: bool):
        self.is_logging = is_logging

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


def logistic_penalty_function(M=1, k=1, b=1, x=0):
    return M / (1 + math.exp(k * (b - x)))


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
    inter_arrival_times = np.random.exponential(1 / rate, int(rate * time_duration) + 100)
    event_times = np.cumsum(inter_arrival_times)
    event_times = event_times[event_times < time_duration]

    return np.floor(event_times).astype(int)


def convert_seconds_to_hours_minutes(seconds) -> str:
    hours, remainder = divmod(seconds, 3600)
    hours = hours % 24
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def truncated_gaussian(lower=5, upper=25, mean=15, std_dev=5):
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    truncated_normal = truncnorm(a, b, loc=mean, scale=std_dev)
    speed = truncated_normal.rvs()
    return speed


def nodes_within_radius(target_node: str, nodes_dict: dict[str, tuple], radius: int) -> dict[str, tuple]:
    # Get coordinates of the target node
    target_coords = nodes_dict.get(target_node)
    if not target_coords:
        raise ValueError("Target node not found in nodes dictionary")

    # Find all nodes within the radius and return as a dictionary
    nearby_nodes = {
        node_id: coords for node_id, coords in nodes_dict.items()
        if node_id != target_node and compute_distance(target_coords, coords) <= radius
    }

    return nearby_nodes

# ----------------------------------------------------------------------------------------------------------------------

def load_cells_from_csv(filename) -> dict[int, "Cell"]:
    from gymnasium_env.simulator.cell import Cell
    df = pd.read_csv(filename)
    cells = {row['id']: Cell.from_dict(row) for _, row in df.iterrows()}
    return cells


def initialize_graph(graph_path: str = None) -> nx.MultiDiGraph:
    """
    Initialize a road graph from a saved file.

    Parameters:
        - graph_path (str):  the file directory

    Returns:
        - The graph representing the road network.
    """
    if os.path.isfile(graph_path):
        print("Network file already exists. Loading the network data: ", end="")
        graph = ox.load_graphml(graph_path)
        print("network data loaded successfully.")
    else:
        # Raise an error if the graph file does not exist
        raise FileNotFoundError("Network file does not exist. Please check the file path.")

    return graph


def initialize_bikes(station: "Station" = None, n: int = 0, next_bike_id: int = 0) -> tuple[dict[int, "Bike"], int]:
    """
    Initialize a list of bikes at a station starting from a specified bike_id.

    Parameters:
        - station (Station): The station where the bikes are going to be initialized.
        - n (int): The number of bikes to initialize. Default is 0.
        - next_bike_id (int): The starting bike_id. Default is 0. 

    Returns:
        - dict: A dictionary containing:
            - dict of bikes at the station
            - the next available bike_id
    """
    from gymnasium_env.simulator.bike import Bike

    next_id = next_bike_id
    bikes = {}
    for i in range(n):
        bike = Bike(station=station, bike_id=next_id)
        next_id += 1
        bikes[bike.get_bike_id()] = bike
        if station is not None:
            station.lock_bike(bike)
    return bikes, next_id


def initialize_stations(stations: dict, depot: dict, bikes_per_station: dict, next_bike_id: int) -> tuple[dict, dict, int]:
    """
    Initialize a list of stations based on the nodes of the graph.

    Parameters:
        - G (nx.MultiDiGraph): The graph representing the road network.

    Returns:
        - dict: A dictionary containing the stations in the network.
    """
    system_bikes = {}

    for station in stations.values():
        station_id = station.get_station_id()
        if station_id != 10000:
            total_bikes_for_station = bikes_per_station.get(station_id)
            bikes = {key: depot.pop(key) for key in list(depot.keys())[:total_bikes_for_station]}
            station.set_bikes(bikes)
            system_bikes.update(bikes)

    outside_system_bikes, next_bike_id = initialize_bikes(n=1000, next_bike_id=next_bike_id)
    for bike in outside_system_bikes.values():
        bike.set_station(stations.get(10000))

    return system_bikes, outside_system_bikes, next_bike_id


def initialize_cells_subgraph(cells: dict[int, "Cell"], nodes_dict: dict[int, tuple[float, float]],
                              distance_matrix: pd.DataFrame, node_features: dict = None) -> nx.MultiDiGraph:
    """
    Initialize a subgraph of the road network based on the cells.

    Parameters:
        - graph (nx.MultiDiGraph): The road network graph.
        - cells (dict): A dictionary of cells in the network.
        - nodes_dict (dict): A dictionary of nodes and their coordinates.

    Returns:
        - nx.Graph: A subgraph of the road network containing the center nodes of the cells.
    """
    # Initialize the subgraph
    subgraph = nx.MultiDiGraph()
    subgraph.graph['crs'] = "EPSG:4326"

    max_length = 0
    nodes_data = {}

    # Default node features if not provided
    default_features = {
        "average_battery_level": 0.0,
        "variance_battery_level": 0.0,
        "low_battery_ratio": 0.0,
        "demand_rate": 0.0,
        "arrival_rate": 0.0,
        "bike_load": 0.0,
        "visits": 0.0,
        "critic_score": 0.0,
    }

    # Use provided features or fall back to defaults
    node_features = node_features or default_features

    # Collect node data and find max length in one pass
    for cell_id, cell in cells.items():
        center_node = cell.get_center_node()
        node_coords = nodes_dict.get(center_node)

        # Initialize node attributes
        # FIXME: change cell_id
        node_attrs = {
            "cell_id": cell.get_id(),
            "x": node_coords[1],
            "y": node_coords[0],
        }
        # Add custom node features
        node_attrs.update(node_features)

        nodes_data[center_node] = node_attrs

        for adjacent_cell_id in cell.get_adjacent_cells().values():
            if adjacent_cell_id and adjacent_cell_id in cells:
                adjacent_center = cells[adjacent_cell_id].get_center_node()
                max_length = max(max_length, distance_matrix.loc[center_node, adjacent_center])

    # Add nodes in bulk
    subgraph.add_nodes_from(nodes_data.items())

    # Add edges
    for cell_id, cell in cells.items():
        current_center = cell.get_center_node()
        for adjacent_cell_id in cell.get_adjacent_cells().values():
            if adjacent_cell_id and adjacent_cell_id in cells:
                adjacent_center = cells[adjacent_cell_id].get_center_node()
                try:
                    path_length = distance_matrix.loc[current_center, adjacent_center]
                    subgraph.add_edge(current_center, adjacent_center, distance=path_length / max_length)
                except nx.NetworkXNoPath:
                    print(f"No path found between {current_center} and {adjacent_center}.")

    return subgraph

# ----------------------------------------------------------------------------------------------------------------------

def plot_graph(graph: nx.MultiDiGraph, cell_dict: dict[int, "Cell"] = None):
    # Extract nodes and edges in WGS84 coordinates (lon, lat)
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the graph edges in geographic coordinates
    edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.7)
    # Plot the graph nodes
    nodes.plot(ax=ax, markersize=2, color="blue", alpha=0.7)

    if cell_dict is not None:
        # Convert cell_dict into a GeoDataFrame in WGS84 for easy plotting
        grid_geoms = [cell.boundary for cell in cell_dict.values()]
        cell_gdf = gpd.GeoDataFrame(geometry=grid_geoms, crs="EPSG:4326")  # WGS84 CRS
        # Overlay the grid cells
        cell_gdf.plot(ax=ax, linewidth=0.8, edgecolor="red", facecolor="blue", alpha=0.2)

    # Configure plot appearance
    plt.axis('off')
    plt.show()


def plot_graph_with_grid(graph, cell_dict, plot_center_nodes=False, plot_number_cells=False, truck_coords=None,
                         x_lim=None, y_lim=None):
    # Extract nodes and edges in WGS84 coordinates (lon, lat)
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    # Convert cell_dict into a GeoDataFrame in WGS84 for easy plotting
    grid_geoms = [cell.boundary for cell in cell_dict.values()]
    cell_gdf = gpd.GeoDataFrame(geometry=grid_geoms, crs="EPSG:4326")  # WGS84 CRS

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the graph edges in geographic coordinates
    edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.5)
    # Plot the graph nodes
    nodes.plot(ax=ax, markersize=2, color="blue", alpha=0.5)

    # Overlay the grid cells
    cell_gdf.plot(ax=ax, linewidth=0.8, edgecolor="red", facecolor="blue", alpha=0.2)

    for cell in cell_dict.values():
        if plot_center_nodes:
            center_node = cell.center_node
            if center_node != 0:
                node_coords = graph.nodes[center_node]['x'], graph.nodes[center_node]['y']
                ax.plot(node_coords[0], node_coords[1], marker='o', color='yellow', markersize=4, label=f"Center Node {cell.id}")

        if plot_number_cells:
            center_coords = cell.boundary.centroid.coords[0]
            ax.text(center_coords[0], center_coords[1], str(cell.id), fontsize=8, color='yellow', ha='center', va='center', weight='bold')

    if truck_coords is not None:
        ax.plot(truck_coords[1], truck_coords[0], marker='o', color='red', markersize=10, label="Truck position")

    if x_lim is not None and y_lim is not None:
        plt.xlim(x_lim)
        plt.ylim(y_lim)

    # Configure plot appearance
    plt.axis('off')
    plt.show()