import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
from scipy.stats import truncnorm
import pandas as pd
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

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
    fig, ax = ox.plot_graph(G, node_size=node_size, node_color=node_color, edge_color=edge_color, edge_linewidth=edge_linewidth, bgcolor=bgcolor, figsize=figsize)
    plt.show()


def plot_graph_with_colored_nodes(graph: nx.MultiDiGraph, rate_matrix: pd.DataFrame):
    """
    Plot the OSMnx graph with nodes colored based on total request rates.

    Parameters:
        - graph: The OSMnx graph.
        - rate_matrix: A matrix containing request rates for each node.
    """
    # Calculate total rates for each starting station
    total_rates = rate_matrix.sum(axis=1)

    # Find minimum and maximum total rates
    min_rate = np.min(total_rates[total_rates > 0])
    max_rate = np.max(total_rates)

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

    # # Create a colorbar and place it inside the plot
    # sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    # sm.set_array([])  # Set an empty array for the ScalarMappable
    # cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, location='right')
    # cbar.set_label('Total Request Rates')

    # Title
    ax.set_title('OSMnx Graph with Node Colors Based on Total Request Rates')

    plt.show()
