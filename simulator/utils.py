import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
import networkx as nx

from scipy.stats import truncnorm
from matplotlib.colors import Normalize
from geopy.distance import distance

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

def plot_poisson_process(data, index):
    """Plot the Poisson process for a given list."""
    plt.plot(data, marker='o', linestyle='-', label=f'Process {index}')
    plt.title(f'Poisson Process {index}')
    plt.xlabel('Time')
    plt.ylabel('Number of Events')
    plt.legend()
    plt.grid()
    plt.show()


def plot_graph(G, node_size=15, node_color="white", edge_color="orange", edge_linewidth=1, bgcolor="black", figsize=(15, 15)):
    _, _ = ox.plot_graph(G, node_size=node_size, node_color=node_color, edge_color=edge_color, edge_linewidth=edge_linewidth, bgcolor=bgcolor, figsize=figsize)
    plt.show()


def plot_graph_with_colored_nodes(graph: nx.MultiDiGraph, rate_matrix: pd.DataFrame, axis: int = 0, colormap: str = None):
    """
    Plot the OSMnx graph with nodes colored based on total request rates.

    Parameters:
        - graph: The OSMnx graph.
        - rate_matrix: A matrix containing request rates for each node.
    """

    if axis == 0:
        sum_array = np.zeros(rate_matrix.shape[0])
        for idx in rate_matrix.index:
            sum_array[idx] = kahan_sum(rate_matrix.loc[idx].values)
    else:
        sum_array = np.zeros(rate_matrix.shape[1])
        for idx in rate_matrix.columns:
            sum_array[int(idx)] = kahan_sum(rate_matrix[idx].values)

    min_rate = sum_array.min()
    max_rate = sum_array.max()

    print(min_rate, max_rate)

    # Normalize total rates to [0, 1] for colormap
    if colormap is not None:
        norm = Normalize(vmin=min_rate, vmax=max_rate)
        colormap = plt.get_cmap(colormap)
        node_colors = {
            node: (colormap(norm(rate_matrix.loc[node].sum()))) if rate_matrix.loc[node].sum() > 0 else (0.3,0.3,0.3,1)  # White color in RGBA
            for node in graph.nodes
        }
    else:
        if axis == 0:
            node_colors = {
                node: (0,1,0,1) if kahan_sum(rate_matrix.loc[node].values) != 0 else (0.3,0.3,0.3,1)
                for node in graph.nodes
            }
        else:
            node_colors = {
                node: (0,1,1,1) if rate_matrix.loc[node].sum() != 0 else (0.3,0.3,0.3,1)
                for node in graph.nodes
            }

    # Plot the graph using OSMnx
    fig, ax = ox.plot_graph(graph, node_color=list(node_colors.values()), node_size=15, edge_color='darkgrey', bgcolor='black', edge_linewidth=0.5, figsize=(15, 15))

    # Title
    ax.set_title('OSMnx Graph with Node Colors Based on Total Request Rates')

    plt.show()
