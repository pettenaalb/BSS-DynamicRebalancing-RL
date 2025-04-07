import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
import geopandas as gpd
import networkx as nx
import calendar

from matplotlib.colors import Normalize
from geopy.distance import great_circle, geodesic, distance
from tqdm import tqdm

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


def haversine_distance(coords1, coords2):
    # Convert degrees to radians
    lat1, lon1 = np.radians(coords1)
    lat2, lon2 = np.radians(coords2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth's radius in meters
    earth_radius = 6371000
    distance_in_meters = earth_radius * c

    return distance_in_meters


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

def plot_graph(graph: nx.MultiDiGraph, path: str = None):
    """
    Plot the OSMnx graph with nodes colored based on total request rates.

    Parameters:
        - graph: The OSMnx graph.
        - rate_matrix: A matrix containing request rates for each node.
    """

    # Plot the graph using OSMnx
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    fig, ax = plt.subplots(figsize=(15, 12), facecolor='white')
    plt.subplots_adjust(left=0, top=1.02, right=1.2, bottom=0, wspace=0, hspace=0)

    # Plot the graph edges in geographic coordinates
    edges.plot(ax=ax, linewidth=0.5, edgecolor="#DC143C", alpha=1, zorder=1)
    # Plot the graph nodes
    nodes.plot(ax=ax, markersize=15, color='#4169E1', alpha=1, zorder=2)

    # Plot specific node
    # node = 64
    # node_coords = (graph.nodes[node]['y'], graph.nodes[node]['x'])
    # ax.plot(node_coords[1], node_coords[0], marker='o', color='red', markersize=4, label=f"Node {node}")

    plt.axis('off')
    plt.savefig(path + 'graph.svg', dpi=300, bbox_inches='tight', pad_inches=0.1)

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
    norm = Normalize(vmin=min_rate, vmax=max_rate)
    if colormap is not None:
        colormap = plt.get_cmap(colormap)
        node_colors = {
            node: (colormap(norm(rate_matrix.loc[node].sum())))
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
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    fig, ax = plt.subplots(figsize=(15, 12), facecolor='black')
    plt.subplots_adjust(left=0, top=1.02, right=1.2, bottom=0, wspace=0, hspace=0)

    # Plot the graph edges in geographic coordinates
    edges.plot(ax=ax, linewidth=0.5, edgecolor="darkgrey", alpha=1, zorder=1)
    # Plot the graph nodes
    nodes['color'] = nodes.index.map(lambda node_id: node_colors.get(node_id))
    nodes.plot(ax=ax, markersize=15, color=nodes['color'], alpha=1, zorder=2)

    if colormap is not None:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])  # Empty array because we don't need data
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.1, pad=0.01)
        cbar.set_label('Request', fontsize=10, color='white')
        cbar.set_ticks([min_rate, max_rate / 2, max_rate])  # Min, 50%, and Max values
        cbar.set_ticklabels([f'Min: {min_rate}', f'50%: {max_rate / 2}', f'Max: {max_rate}'])

        cbar.ax.tick_params(axis='y', colors='white')  # Ticks color
        cbar.ax.yaxis.set_tick_params(labelcolor='white')  # Tick labels color
        # Positioning the colorbar in the upper right corner
        cbar.ax.yaxis.set_label_position('right')  # Position the label on the left of the colorbar
        cbar.ax.yaxis.set_ticks_position('right')  # Position the ticks on the left side
        cbar.ax.set_position([1.0-0.2, 1-0.12, 0.07, 0.1])  # Adjust position (x, y, width, height)

    plt.axis('off')
    plt.show()


def plot_graph_with_grid(graph, cell_dict, plot_center_nodes=False, plot_number_cells=False):
    # Extract nodes and edges in WGS84 coordinates (lon, lat)
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    # Convert cell_dict into a GeoDataFrame in WGS84 for easy plotting
    grid_geoms = [cell.boundary for cell in cell_dict.values()]
    cell_gdf = gpd.GeoDataFrame(geometry=grid_geoms, crs="EPSG:4326")  # WGS84 CRS

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the graph edges in geographic coordinates
    edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.7)
    # Plot the graph nodes
    nodes.plot(ax=ax, markersize=2, color="blue", alpha=0.7)

    # Overlay the grid cells
    cell_gdf.plot(ax=ax, linewidth=0.8, edgecolor="red", facecolor="blue", alpha=0.5)

    for cell in cell_dict.values():
        center_node = cell.center_node
        if center_node != 0:
            node_coords = graph.nodes[center_node]['x'], graph.nodes[center_node]['y']

            if plot_center_nodes:
                ax.plot(node_coords[0], node_coords[1], marker='o', color='yellow', markersize=4,
                        label=f"Center Node {cell.id}")

            # Connect to adjacent cells' center nodes
            for direction, adjacent_cell in cell.adjacent_cells.items():
                if adjacent_cell is not None and adjacent_cell in cell_dict:
                    adjacent_center_node = cell_dict[adjacent_cell].center_node
                    if adjacent_center_node != 0:
                        adj_coords = graph.nodes[adjacent_center_node]['x'], graph.nodes[adjacent_center_node]['y']
                        ax.plot([node_coords[0], adj_coords[0]], [node_coords[1], adj_coords[1]], color='yellow',
                                linewidth=1.5, alpha=0.8)

        if plot_number_cells:
            center_coords = cell.boundary.centroid.coords[0]
            ax.text(center_coords[0], center_coords[1], str(cell.id), fontsize=8, color='yellow', ha='center',
                    va='center', weight='bold')

    # Configure plot appearance
    plt.axis('off')
    plt.savefig('../data_new/grid.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)

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
        if node_id != target_node and haversine_distance(target_coords, coords) <= radius
    }

    return nearby_nodes