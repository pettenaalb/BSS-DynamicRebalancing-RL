import os
import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point


params = {
    'place': ["Cambridge, Massachusetts, USA"],
    'network_type': 'bike',
    'data_path': "../data/",
    'graph_file': "cambridge_network.graphml",
}

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

# ----------------------------------------------------------------------------------------------------------------------

class Cell:
    def __init__(self, cell_id, min_x, min_y, max_x, max_y):
        self.id = cell_id
        self.boundary = box(min_x, min_y, max_x, max_y)
        self.nodes = []


def divide_graph_into_cells(graph: nx.MultiDiGraph, cell_size: int) -> dict[int, Cell]:
    """
    Divides the graph into cells of a given size.

    Parameters:
        - graph (nx.MultiDiGraph): The graph object representing the street network.
        - cell_size (int): The size of each cell in meters.

    Returns:
        - cell_dict (dict): A dictionary containing the cell objects.
    """
    nodes = ox.graph_to_gdfs(graph, edges=False)
    min_x, min_y, max_x, max_y = nodes.total_bounds

    # Conversion factors
    meters_per_degree_lat = 111320  # approximately, for latitude
    meters_per_degree_lon = 111320 * abs(np.cos(np.radians((min_y + max_y) / 2)))  # adjusted for longitude based on latitude

    # Convert cell size from meters to degrees
    cell_size_x_deg = cell_size / meters_per_degree_lon
    cell_size_y_deg = cell_size / meters_per_degree_lat

    num_x_cells = int((max_x - min_x) / cell_size_x_deg) + 1
    num_y_cells = int((max_y - min_y) / cell_size_y_deg) + 1

    cell_dict = {}
    cell_id = 0
    for i in range(num_x_cells):
        for j in range(num_y_cells):
            cell_min_x = min_x + i * cell_size_x_deg
            cell_min_y = min_y + j * cell_size_y_deg
            cell_max_x = cell_min_x + cell_size_x_deg
            cell_max_y = cell_min_y + cell_size_y_deg

            cell = Cell(cell_id, cell_min_x, cell_min_y, cell_max_x, cell_max_y)
            cell_dict[cell_id] = cell
            cell_id += 1

    return cell_dict


def assign_nodes_to_cells(graph: nx.MultiDiGraph, cell_dict: dict[int, Cell]) -> dict[int, int]:
    """
    Assigns nodes to the corresponding cells based on their coordinates.

    Parameters:
        - graph (nx.MultiDiGraph): The graph object representing the street network.
        - cell_dict (dict): A dictionary containing the cell objects.

    Returns:
        - node_dict (dict): A dictionary containing the mapping of nodes to cells.
    """
    nodes = ox.graph_to_gdfs(graph, edges=False)

    node_dict = {}
    for node_id, row in nodes.iterrows():
        node_point = Point(row["x"], row["y"])

        for cell in cell_dict.values():
            if cell.boundary.contains(node_point):
                cell.nodes.append(node_id)
                node_dict[node_id] = cell.id
                break

    return node_dict

# ----------------------------------------------------------------------------------------------------------------------

def save_cells_to_csv(cell_dict: dict[int, Cell], file_path: str):
    """
    Saves cell information to a CSV file using a pandas DataFrame.
    Each row will contain: cell_id, min_x, min_y, max_x, max_y

    Parameters:
        - cell_dict (dict): A dictionary containing the cell objects.
        - file_path (str): The path to the CSV file where the data will be saved.
    """
    cell_data = {
        "cell_id": [],
        "min_x": [],
        "min_y": [],
        "max_x": [],
        "max_y": [],
        "num_nodes": []
    }
    for cell_id, cell in cell_dict.items():
        if len(cell.nodes) != 0:
            bounds = cell.boundary.bounds
            cell_data["cell_id"].append(cell_id)
            cell_data["min_x"].append(bounds[0])
            cell_data["min_y"].append(bounds[1])
            cell_data["max_x"].append(bounds[2])
            cell_data["max_y"].append(bounds[3])
            cell_data["num_nodes"].append(len(cell.nodes))

    cell_df = pd.DataFrame(cell_data).set_index("cell_id")
    cell_df.to_csv(file_path)


def load_cells_from_csv(file_path: str) -> dict[int, Cell]:
    """
    Loads cell information from a CSV file and reconstructs cell_dict.
    """
    cell_df = pd.read_csv(file_path, index_col="cell_id")
    cell_dict = {}
    for cell_id, row in cell_df.iterrows():
        cell = Cell(cell_id, row["min_x"], row["min_y"], row["max_x"], row["max_y"])
        cell_dict[cell_id] = cell
    return cell_dict


def save_node_cell_map_to_csv(node_dict: dict[int, int], file_path: str):
    """
    Saves the node-cell mapping to a CSV file using a pandas DataFrame.
    Each row will contain: node_id, cell_id

    Parameters:
        - node_dict (dict): A dictionary containing the mapping of nodes to cells.
        - file_path (str): The path to the CSV file where the data will be saved.
    """
    node_data = {
        "osmid": [],
        "cell_id": []
    }
    for node_id, cell_id in node_dict.items():
        node_data["osmid"].append(node_id)
        node_data["cell_id"].append(cell_id)

    node_df = pd.DataFrame(node_data).set_index("osmid")
    node_df.to_csv(file_path)


def load_node_cell_map_from_csv(cell_dict: dict[int, Cell], file_path: str) -> dict[int, int]:
    """
    Loads node-cell mappings from a CSV file and reconstructs node_cell_map.
    """
    node_df = pd.read_csv(file_path, index_col="osmid")
    node_dict = {}
    for node_id, row in node_df.iterrows():
        node_dict[node_id] = row["cell_id"]
        cell_dict[row["cell_id"]].nodes.append(node_id)

    return node_dict

# ----------------------------------------------------------------------------------------------------------------------

def plot_graph_with_grid(graph, cell_dict):
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
    cell_gdf.boundary.plot(ax=ax, linewidth=0.8, edgecolor="red", alpha=0.5)

    # Configure plot appearance
    ax.set_title("OSMnx Graph with Grid Overlay (Lon, Lat)", fontsize=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal', 'box')  # keep aspect ratio equal
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------

def main():
    # Initialize the graph
    print("Initializing the graph... ")
    graph = initialize_graph(params['data_path'] + params['graph_file'])

    # Divide the graph into cells
    print("Dividing the graph into cells... ")
    cell_dict = divide_graph_into_cells(graph, 300)

    # Assign nodes to cells
    print("Assigning nodes to cells... ")
    node_dict = assign_nodes_to_cells(graph, cell_dict)

    # Save the cell information to a CSV file
    print("Saving cell information to CSV... ")
    save_cells_to_csv(cell_dict, params['data_path'] + "cell_data.csv")

    # Save the node-cell mapping to a CSV file
    print("Saving node-cell mapping to CSV... ")
    save_node_cell_map_to_csv(node_dict, params['data_path'] + "node_cell_map.csv")

    plot_graph_with_grid(graph, cell_dict)


if __name__ == '__main__':
    main()