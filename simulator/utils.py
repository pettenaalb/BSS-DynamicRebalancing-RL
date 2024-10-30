import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
from scipy.stats import truncnorm

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