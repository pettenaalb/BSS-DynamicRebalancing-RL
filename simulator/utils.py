import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
from scipy.stats import truncnorm
import pandas as pd
import networkx as nx
from matplotlib.colors import Normalize
import calendar
from geopy.distance import great_circle
from tqdm import tqdm

def generate_poisson_events(rate, time_duration) -> list[int]:
    """
    Generate Poisson events within a specified time duration.

    Parameters:
        - rate (float): The average rate of events per unit time.
        - time_duration (float): The total time duration in which events can occur.

    Returns:
        - list: A list of event times occurring within the specified time duration.
    """
    num_events = np.random.poisson(rate * time_duration)
    inter_arrival_times = np.random.exponential(1.0 / rate, num_events)
    event_times = np.cumsum(inter_arrival_times)
    valid_event_times = event_times[event_times <= time_duration]

    return valid_event_times.astype(int).tolist()


def convert_seconds_to_hours_minutes(seconds) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}"


def truncated_gaussian_speed(lower=5, upper=25, mean=15, std_dev=5):
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    truncated_normal = truncnorm(a, b, loc=mean, scale=std_dev)

    speed = truncated_normal.rvs()

    return speed


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


def kahan_sum(arr):
    total = 0.0
    c = 0.0  # A running compensation for lost low-order bits.
    for value in arr:
        y = value - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


def plot_graph_with_colored_nodes(graph: nx.MultiDiGraph, rate_matrix: pd.DataFrame):
    """
    Plot the OSMnx graph with nodes colored based on total request rates.

    Parameters:
        - graph: The OSMnx graph.
        - rate_matrix: A matrix containing request rates for each node.
    """
    row_sums_array = np.array([kahan_sum(rate_matrix.iloc[i].values) for i in range(rate_matrix.shape[0])])

    min_rate = row_sums_array.min()
    max_rate = row_sums_array.max()

    print(min_rate, max_rate)

    # Normalize total rates to [0, 1] for colormap
    norm = Normalize(vmin=min_rate, vmax=max_rate)
    colormap = plt.get_cmap('coolwarm')  # Choose a colormap

    # Create a color dictionary for nodes
    # Assuming `colormap` and `norm` are defined, and `total_rates` is available
    node_colors = {
        node: (colormap(norm(rate_matrix.loc[node].sum()))) if rate_matrix.loc[node].sum() > 0 else (0.3,0.3,0.3,1)  # White color in RGBA
        for node in graph.nodes
    }

    # Plot the graph using OSMnx
    fig, ax = ox.plot_graph(graph, node_color=list(node_colors.values()), node_size=15, edge_color='darkgrey', bgcolor='black', edge_linewidth=0.5, figsize=(15, 15))

    # Title
    ax.set_title('OSMnx Graph with Node Colors Based on Total Request Rates')

    plt.show()


def count_specific_day(year: int, month: int, day_name: str) -> int:
    # Mapping of day names to weekday numbers
    day_map = {
        'monday': 0,
        'tuesday': 1,
        'wednesday': 2,
        'thursday': 3,
        'friday': 4,
        'saturday': 5,
        'sunday': 6
    }

    # Get the day number for the specified day name
    if day_name.lower() not in day_map:
        raise ValueError("Invalid day name. Choose from: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.")

    day_number = day_map[day_name.lower()]

    # Get the number of days in the month
    num_days = calendar.monthrange(year, month)[1]

    # Count how many times the specified day occurs in that month
    count = 0
    for day in range(1, num_days + 1):
        if calendar.weekday(year, month, day) == day_number:
            count += 1

    return count





def find_nearby_nodes(graph: nx.MultiDiGraph, target_node: int, radius_meters: float) -> list[int]:
    """
    Find nearby nodes within a specified radius around the target node.

    Parameters:
        - graph (nx.MultiDiGraph): The graph representing the road network.
        - target_node (int): The target node to find nearby nodes around.
        - radius_meters (float): The radius in meters within which to find nearby nodes.

    Returns:
        - list: A list of node IDs that are within the specified radius around the target
    """
    # Check if the target node is in the graph
    if target_node not in graph:
        raise ValueError(f"Node {target_node} is not in the graph.")

    target_coords = (graph.nodes[target_node]['y'], graph.nodes[target_node]['x'])
    nearby_nodes = []

    # Find nearby nodes within the specified radius
    for node in graph.nodes:
        if node != target_node:
            node_coords = (graph.nodes[node]['y'], graph.nodes[node]['x'])
            distance = great_circle(target_coords, node_coords).meters

            if distance <= radius_meters:
                nearby_nodes.append(node)

    return nearby_nodes


def connect_disconnected_neighbors(graph: nx.MultiDiGraph, radius_meters: int):
    """
    Connect disconnected nodes in the graph by adding edges between them.

    Parameters:
        - graph (nx.MultiDiGraph): The graph representing the road network.
        - radius_meters (int): The radius in meters within which to connect disconnected nodes.
    """

    tbar = tqdm(total=len(graph.nodes), desc="Connecting disconnected nodes")

    for node in graph.nodes:
        # Check if the node has valid coordinates
        if 'y' not in graph.nodes[node] or 'x' not in graph.nodes[node]:
            print(f"Node {node} does not have valid coordinates.")
            continue

        # Find nearby nodes within the specified radius
        nearby_nodes = find_nearby_nodes(graph, node, radius_meters)

        # Add an edge between the node and its neighbors if they are not already connected
        for neighbor in nearby_nodes:
            if not nx.has_path(graph, node, neighbor):
                node_coords = (graph.nodes[node]['y'], graph.nodes[node]['x'])
                neighbor_coords = (graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x'])

                if 'y' not in graph.nodes[neighbor] or 'x' not in graph.nodes[neighbor]:
                    print(f"Neighbor {neighbor} does not have valid coordinates.")
                    continue
                distance_meters = great_circle(node_coords, neighbor_coords).meters

                speed_kph = 15.0
                travel_time_hours = distance_meters / 1000 / speed_kph
                weight = travel_time_hours * 3600
                graph.add_edge(node, neighbor, length=distance_meters, speed_kph=speed_kph, weight=weight)

        tbar.update(1)
