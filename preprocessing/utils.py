import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
import networkx as nx
import calendar

from matplotlib.colors import Normalize
from geopy.distance import great_circle, geodesic
from tqdm import tqdm
from math import radians, cos, sin, sqrt, atan2

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


def haversine(coords1, coords2):
    # Radius of Earth in meters
    R = 6371000
    lat1, lon1 = radians(coords1[0]), radians(coords1[1])
    lat2, lon2 = radians(coords2[0]), radians(coords2[1])

    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    a = sin(d_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(d_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


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

# ----------------------------------------------------------------------------------------------------------------------

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

    min_rate = 0
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

# ----------------------------------------------------------------------------------------------------------------------

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


def maximum_distance_between_points(G: nx.MultiDiGraph) -> int:
    """
    Compute the maximum distance between any two nodes in the graph.

    Parameters:
        - G (networkx.MultiDiGraph): The graph representing the road network.

    Returns:
        - float: The maximum distance between any two nodes in the graph.
    """
    max_distance = 0
    for u, v, data in G.edges(data=True):
        # Get coordinates of the nodes
        u_coords = (G.nodes[u]['y'], G.nodes[u]['x'])
        v_coords = (G.nodes[v]['y'], G.nodes[v]['x'])

        # Calculate the distance between the nodes
        distance = geodesic(u_coords, v_coords).meters

        # Update maximum distance and edge if current distance is greater
        if distance > max_distance:
            max_distance = distance
    return max_distance


def is_within_graph_bounds(G: nx.MultiDiGraph, node_coords: tuple, nearest_node, threshold=500) -> bool:
    """
    Check if a point is within the bounds of the graph.

    Parameters:
        - G (networkx.MultiDiGraph): The graph representing the road network.
        - lat (float): The latitude of the point.
        - lon (float): The longitude of the point.
        - threshold (int): The maximum distance allowed between the point and the nearest node in the graph.

    Returns:
        - bool: True if the point is within the bounds of the graph, False otherwise.
    """
    # Find the nearest node to the point in the graph
    nearest_node_coords = (G.nodes[nearest_node]['y'], G.nodes[nearest_node]['x'])

    # Compute the distance between the point and the nearest node
    distance_to_nearest_node = geodesic((node_coords[0], node_coords[1]), nearest_node_coords).meters

    # Check if this distance is within the acceptable threshold
    return distance_to_nearest_node <= threshold


def nodes_within_radius(target_node: str, nodes_dict: dict[str, tuple], radius: int) -> dict[str, tuple]:
    # Get coordinates of the target node
    target_coords = nodes_dict.get(target_node)
    if not target_coords:
        raise ValueError("Target node not found in nodes dictionary")

    # Find all nodes within the radius and return as a dictionary
    nearby_nodes = {
        node_id: coords for node_id, coords in nodes_dict.items()
        if node_id != target_node and haversine(target_coords, coords) <= radius
    }

    return nearby_nodes