import os
import pandas as pd
import osmnx as ox
import networkx as nx
import logging

from simulator.cell import Cell
from simulator.utils import truncated_gaussian
from simulator.truck import Truck
from simulator.station import Station
from simulator.bike_simulator import initialize_stations
from simulator.bike import Bike

params = {
    'data_path': "../data/",
    'graph_file': "cambridge_network.graphml",
    'network_type': 'drive',

    'year': 2022,
    'month': [9, 10],
    'day': "monday",
    'time_slot': 3,
    'time_interval': 3600*3   # 3 hour
}

system_bikes = {}
outside_system_bikes = {}

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


def load_cells_from_csv(filename) -> dict[int, Cell]:
    df = pd.read_csv(filename)
    cells = {row['id']: Cell.from_dict(row) for _, row in df.iterrows()}
    return cells

# ----------------------------------------------------------------------------------------------------------------------

def move_up(truck: Truck, distance_matrix: pd.DataFrame, cell_dict: dict[int, Cell]) -> int:
    cell = truck.get_cell()
    up_cell = cell_dict.get(cell.get_adjacent_cells().get('up'))

    if up_cell is None:
        return 0

    distance = distance_matrix.loc[truck.get_position(), up_cell.get_center_node()]
    velocity_kmh = truncated_gaussian(10, 70, 40, 5)
    time = int(distance * 3.6 / velocity_kmh)

    truck.set_position(up_cell.get_center_node())
    truck.set_range(truck.get_range() - distance)
    truck.set_cell(up_cell)
    truck.set_range(truck.max_range - distance/1000)

    return time


def move_down(truck: Truck, distance_matrix: pd.DataFrame, cell_dict: dict[int, Cell]) -> int:
    cell = truck.get_cell()
    down_cell = cell_dict.get(cell.get_adjacent_cells().get('down'))

    if down_cell is None:
        return 0

    distance = distance_matrix.loc[truck.get_position(), down_cell.get_center_node()]
    velocity_kmh = truncated_gaussian(10, 70, 40, 5)
    time = int(distance * 3.6 / velocity_kmh)

    truck.set_position(down_cell.get_center_node())
    truck.set_range(truck.get_range() - distance)
    truck.set_cell(down_cell)
    truck.set_range(truck.max_range - distance/1000)

    return time


def move_left(truck: Truck, distance_matrix: pd.DataFrame, cell_dict: dict[int, Cell]) -> int:
    cell = truck.get_cell()
    left_cell = cell_dict.get(cell.get_adjacent_cells().get('left'))

    if left_cell is None:
        return 0

    distance = distance_matrix.loc[truck.get_position(), left_cell.get_center_node()]
    velocity_kmh = truncated_gaussian(10, 70, 40, 5)
    time = int(distance * 3.6 / velocity_kmh)

    truck.set_position(left_cell.get_center_node())
    truck.set_range(truck.get_range() - distance)
    truck.set_cell(left_cell)
    truck.set_range(truck.max_range - distance/1000)

    return time


def move_right(truck: Truck, distance_matrix: pd.DataFrame, cell_dict: dict[int, Cell]) -> int:
    cell = truck.get_cell()
    right_cell = cell_dict.get(cell.get_adjacent_cells().get('right'))

    if right_cell is None:
        return 0

    distance = distance_matrix.loc[truck.get_position(), right_cell.get_center_node()]
    velocity_kmh = truncated_gaussian(10, 70, 40, 5)
    time = int(distance * 3.6 / velocity_kmh)

    truck.set_position(right_cell.get_center_node())
    truck.set_range(truck.get_range() - distance)
    truck.set_cell(right_cell)
    truck.set_range(truck.max_range - distance/1000)

    return time


def drop_bike(truck: Truck, station_dict: dict[int, Station], distance_matrix: pd.DataFrame) -> int:
    position = truck.get_position()
    center_cell_position = truck.get_cell().get_center_node()

    time = 0
    if position != center_cell_position:
        distance = distance_matrix.loc[position, center_cell_position]
        velocity_kmh = truncated_gaussian(10, 70, 40, 5)
        time = int(distance * 3.6 / velocity_kmh)
        truck.set_position(center_cell_position)

    station = station_dict.get(truck.get_position())
    station.lock_bike(truck.unload_bike())

    return time


def pick_up_bike(truck: Truck, station_dict: dict[int, Station], distance_matrix: pd.DataFrame) -> int:
    cell = truck.get_cell()
    nearby_stations = {station_id: station_dict.get(station_id) for station_id in cell.get_nodes()}
    bike_dict = {}
    for station in nearby_stations.values():
        bike_dict.update(station.get_bikes())

    # Find max distance between truck and nearby station
    max_distance = cell.get_diagonal()
    bikes_metric = {}
    for bike_id, bike in bike_dict.items():
        distance = distance_matrix.loc[truck.get_position(), bike.get_station().get_station_id()]
        if distance > max_distance:
            max_distance = distance
        battery = bike.get_battery() / bike.get_max_battery()
        if distance != 0:
            bikes_metric[bike_id] = (distance/max_distance) * (1 - battery)
        else:
            bikes_metric[bike_id] = (1 - battery)

    # Find the lowest metric bike
    bike_id = min(bikes_metric, key=bikes_metric.get)

    station = bike_dict[bike_id].get_station()

    distance = distance_matrix.loc[truck.get_position(), station.get_station_id()]
    velocity_kmh = truncated_gaussian(10, 70, 40, 5)
    time = int(distance * 3.6 / velocity_kmh)

    bike = station.unlock_bike(bike_id)
    truck.load_bike(bike)

    return time


def stay() -> int: return 0


def charge_bike(truck: Truck, station_dict: dict[int, Station], distance_matrix: pd.DataFrame) -> int:
    t1 = pick_up_bike(truck, station_dict, distance_matrix)

    station = station_dict.get(truck.get_position())

    bike = truck.unload_bike()
    station.lock_bike(bike)

    return t1

# ----------------------------------------------------------------------------------------------------------------------

def main():
    ox.settings.use_cache = True
    logging.basicConfig(filename='trip_output.log', level=logging.INFO, filemode='w')

    # graph = initialize_graph(params['data_path'] + params['graph_file'])

    cells_dict = load_cells_from_csv(params['data_path'] + 'cell-data.csv')

    # Initialize the graph
    print("Initializing the graph... ")
    graph = initialize_graph(params['data_path'] + params['graph_file'])

    # Initialize distance matrix
    distance_matrix = pd.read_csv(params['data_path'] + '/distance-matrix.csv', index_col=0)
    distance_matrix.index = distance_matrix.index.astype(int)
    distance_matrix.columns = distance_matrix.columns.astype(int)

    # Initialize stations
    print("Initializing stations... ")
    global system_bikes
    stations, system_bikes = initialize_stations(graph)

    # Initialize truck
    cell = cells_dict[185]
    bikes = {}
    for i in range(30):
        bikes[i] = Bike()
    truck = Truck(cell.center_node, cell, max_range=300, max_load=30, bikes=bikes)

    print(f"\nMoving up from cell {truck.get_cell().get_id()}")
    time = move_up(truck, distance_matrix, cells_dict)
    print(f"Truck moved up in {time} seconds to cell {truck.get_cell().get_id()}")
    print(f"Truck info: {truck}")

    print(f"\nMoving left from cell {truck.get_cell().get_id()}")
    time = move_left(truck, distance_matrix, cells_dict)
    print(f"Truck moved up in {time} seconds to cell {truck.get_cell().get_id()}")
    print(f"Truck info: {truck}")

    print(f"\nMoving down from cell {truck.get_cell().get_id()}")
    time = move_down(truck, distance_matrix, cells_dict)
    print(f"Truck moved up in {time} seconds to cell {truck.get_cell().get_id()}")
    print(f"Truck info: {truck}")

    print(f"\nMoving right from cell {truck.get_cell().get_id()}")
    time = move_right(truck, distance_matrix, cells_dict)
    print(f"Truck moved up in {time} seconds to cell {truck.get_cell().get_id()}")
    print(f"Truck info: {truck}")

    print(f"\nDropping bike from cell {truck.get_cell().get_id()}")
    time = drop_bike(truck, stations, distance_matrix)
    print(f"Truck moved up in {time} seconds to cell {truck.get_cell().get_id()}")
    print(f"Truck info: {truck}")

    print(f"\nPicking up bike from cell {truck.get_cell().get_id()}")
    time = pick_up_bike(truck, stations, distance_matrix)
    print(f"Truck moved up in {time} seconds to cell {truck.get_cell().get_id()}")
    print(f"Truck info: {truck}")

    print(f"\nStaying at cell {truck.get_cell().get_id()}")
    time = stay()
    print(f"Truck moved up in {time} seconds to cell {truck.get_cell().get_id()}")



if __name__ == '__main__':
    main()