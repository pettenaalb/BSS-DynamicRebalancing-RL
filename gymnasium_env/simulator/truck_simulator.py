import pandas as pd

from gymnasium_env.simulator.cell import Cell
from gymnasium_env.simulator.utils import truncated_gaussian
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.station import Station
from gymnasium_env.simulator.bike import Bike

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
    truck.set_cell(up_cell)

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
    truck.set_cell(down_cell)

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
    truck.set_cell(left_cell)

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
    truck.set_cell(right_cell)

    return time


def drop_bike(truck: Truck, distance_matrix: pd.DataFrame) -> int:
    position = truck.get_position()
    center_cell_position = truck.get_cell().get_center_node()

    time = 0
    if position != center_cell_position:
        distance = distance_matrix.loc[position, center_cell_position]
        velocity_kmh = truncated_gaussian(10, 70, 40, 5)
        time = int(distance * 3.6 / velocity_kmh)
        truck.set_position(center_cell_position)

    return time


def pick_up_bike(truck: Truck, station_dict: dict[int, Station], distance_matrix: pd.DataFrame) -> int:
    cell = truck.get_cell()
    nearby_stations = {station_id: station_dict.get(station_id) for station_id in cell.get_nodes()}
    bike_dict = {}
    for station in nearby_stations.values():
        bike_dict.update(station.get_bikes())

    if len(bike_dict) == 0:
        return 0

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
    truck.set_position(station.get_station_id())

    return time


def stay() -> int: return 0


def charge_bike(truck: Truck, station_dict: dict[int, Station], distance_matrix: pd.DataFrame) -> int:
    return pick_up_bike(truck, station_dict, distance_matrix)
