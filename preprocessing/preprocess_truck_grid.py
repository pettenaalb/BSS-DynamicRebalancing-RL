import os
import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx

from shapely.geometry import Polygon, Point
from tqdm import tqdm
from utils import compute_distance, plot_graph_with_grid
from simulator.cell import Cell

params = {
    'place': ["Cambridge, Massachusetts, USA"],
    'network_type': 'bike',
    'data_path': "../data/",
    'graph_file': "utils/cambridge_network.graphml",
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

def divide_graph_into_cells(graph: nx.MultiDiGraph, cell_size: int) -> dict[int, Cell]:
    """
    Divides the graph into cells of a given size and returns a dictionary containing the cell objects.

    Parameters:
        - graph (nx.MultiDiGraph): The graph object representing the street network.
        - cell_size (int): The size of the cell in meters.

    Returns:
        - cell_dict (dict): A dictionary containing the cell objects.
    """
    nodes = ox.graph_to_gdfs(graph, edges=False)
    min_x, min_y, max_x, max_y = nodes.total_bounds

    # Conversion factors
    lat_deg_to_meter: float = 111320  # 1 degree latitude = 111320 meters
    lat_lon_to_meter: float = 111320 * abs(np.cos(np.radians(np.mean([min_y, max_y]))))

    x_diagonal_deg = cell_size * np.sqrt(2) / lat_lon_to_meter
    y_diagonal_deg = cell_size * np.sqrt(2) / lat_deg_to_meter

    x_centers_od = np.arange(min_x, max_x, x_diagonal_deg)
    x_centers_even = np.arange(min_x + x_diagonal_deg/2, max_x, x_diagonal_deg)
    y_centers_od = np.arange(min_y, max_y + y_diagonal_deg, y_diagonal_deg)
    y_centers_even = np.arange(min_y + y_diagonal_deg/2, max_y, y_diagonal_deg)

    cell_dict = {}
    cell_id = 0
    for x_centers, y_centers in zip([x_centers_od, x_centers_even], [y_centers_od, y_centers_even]):
        for x in x_centers:
            for y in y_centers:
                vertices = [
                    (x + x_diagonal_deg/2, y),
                    (x, y + y_diagonal_deg/2),
                    (x - x_diagonal_deg/2, y),
                    (x, y - y_diagonal_deg/2)
                ]
                cell_boundary = Polygon(vertices)
                cell = Cell(cell_id, cell_boundary)
                cell_dict[cell_id] = cell
                cell_id += 1

    return cell_dict


def assign_nodes_to_cells(graph: nx.MultiDiGraph, cell_dict: dict[int, Cell]) -> list[tuple]:
    """
    Assigns nodes to the corresponding cells based on their coordinates.

    Parameters:
        - graph (nx.MultiDiGraph): The graph object representing the street network.
        - cell_dict (dict): A dictionary containing the cell objects.

    Returns:
        - node_dict (dict): A dictionary containing the mapping of nodes to cells.
    """
    nodes = ox.graph_to_gdfs(graph, edges=False)

    tbar = tqdm(total=len(nodes), desc="Assigning nodes to cells", dynamic_ncols=True)
    node_dict = [tuple]
    for node_id, row in nodes.iterrows():
        node_point = Point(row["x"], row["y"])

        # Check each cell to see if the node point is within its boundary
        for cell in cell_dict.values():
            if cell.boundary.contains(node_point):
                cell.nodes.append(node_id)
                node_dict.append((node_id, cell.id))
                break

        tbar.update(1)

    return node_dict


def set_adjacent_cells(cell_dict: dict[int, Cell]):
    tbar = tqdm(total=len(cell_dict), desc="Setting adjacent cells", dynamic_ncols=True)
    for cell in cell_dict.values():
        center_coords = cell.boundary.centroid.coords[0]
        for adj_cell in cell_dict.values():
            if adj_cell.id != cell.id:
                adj_center_coords = adj_cell.boundary.centroid.coords[0]
                if compute_distance(center_coords, adj_center_coords) < 300:
                    lon_diff = center_coords[0] - adj_center_coords[0]
                    lat_diff = center_coords[1] - adj_center_coords[1]

                    if lon_diff > 0 and lat_diff > 0:
                        cell.adjacent_cells['left'] = adj_cell.id
                        adj_cell.adjacent_cells['right'] = cell.id

                    if lon_diff < 0 and lat_diff < 0:
                        cell.adjacent_cells['right'] = adj_cell.id
                        adj_cell.adjacent_cells['left'] = cell.id

                    if lon_diff > 0 > lat_diff:
                        cell.adjacent_cells['up'] = adj_cell.id
                        adj_cell.adjacent_cells['down'] = cell.id

                    if lon_diff < 0 < lat_diff:
                        cell.adjacent_cells['down'] = adj_cell.id
                        adj_cell.adjacent_cells['up'] = cell.id

        tbar.update(1)

# ----------------------------------------------------------------------------------------------------------------------

def save_cells_to_csv(cells, filename):
    cell_dicts = [cell.to_dict() for cell in cells]
    df = pd.DataFrame(cell_dicts)
    df.to_csv(filename, index=False)

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
    assign_nodes_to_cells(graph, cell_dict)

    cell_to_remove = []
    for cell in cell_dict.values():
        if len(cell.nodes) == 0:
            cell_to_remove.append(cell.id)

    for cell_id in cell_to_remove:
        cell_dict.pop(cell_id)

    for cell in cell_dict.values():
        cell.set_center_node(graph)
        cell.set_diagonal()

    set_adjacent_cells(cell_dict)

    plot_graph_with_grid(graph, cell_dict, plot_number_cells=True)

    plot_graph_with_grid(graph, cell_dict, plot_center_nodes=True)

    save_cells_to_csv(cell_dict.values(), params['data_path'] + "cell-data.csv")


if __name__ == '__main__':
    main()