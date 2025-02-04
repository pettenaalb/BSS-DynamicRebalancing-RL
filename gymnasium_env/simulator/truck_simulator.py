from types import NoneType

import pandas as pd

from gymnasium_env.simulator.cell import Cell
from gymnasium_env.simulator.utils import truncated_gaussian
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.station import Station

# ----------------------------------------------------------------------------------------------------------------------

def move_up(truck: Truck, distance_matrix: pd.DataFrame, cell_dict: dict[int, Cell], mean_velocity: int) -> tuple[int, int]:
    cell = truck.get_cell()
    up_cell = cell_dict.get(cell.get_adjacent_cells().get('up'))

    if up_cell is None:
        return 0, 0

    distance = distance_matrix.loc[truck.get_position(), up_cell.get_center_node()]
    velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
    time = int(distance * 3.6 / velocity_kmh)

    truck.set_position(up_cell.get_center_node())
    truck.set_cell(up_cell)

    return time, distance


def move_down(truck: Truck, distance_matrix: pd.DataFrame, cell_dict: dict[int, Cell], mean_velocity: int) -> tuple[int, int]:
    cell = truck.get_cell()
    down_cell = cell_dict.get(cell.get_adjacent_cells().get('down'))

    if down_cell is None:
        return 0, 0

    distance = distance_matrix.loc[truck.get_position(), down_cell.get_center_node()]
    velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
    time = int(distance * 3.6 / velocity_kmh)

    truck.set_position(down_cell.get_center_node())
    truck.set_cell(down_cell)

    return time, distance


def move_left(truck: Truck, distance_matrix: pd.DataFrame, cell_dict: dict[int, Cell], mean_velocity: int) -> tuple[int, int]:
    cell = truck.get_cell()
    left_cell = cell_dict.get(cell.get_adjacent_cells().get('left'))

    if left_cell is None:
        return 0, 0

    distance = distance_matrix.loc[truck.get_position(), left_cell.get_center_node()]
    velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
    time = int(distance * 3.6 / velocity_kmh)

    truck.set_position(left_cell.get_center_node())
    truck.set_cell(left_cell)

    return time, distance


def move_right(truck: Truck, distance_matrix: pd.DataFrame, cell_dict: dict[int, Cell], mean_velocity: int) -> tuple[int, int]:
    cell = truck.get_cell()
    right_cell = cell_dict.get(cell.get_adjacent_cells().get('right'))

    if right_cell is None:
        return 0, 0

    distance = distance_matrix.loc[truck.get_position(), right_cell.get_center_node()]
    velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
    time = int(distance * 3.6 / velocity_kmh)

    truck.set_position(right_cell.get_center_node())
    truck.set_cell(right_cell)

    return time, distance


def drop_bike(truck: Truck, distance_matrix: pd.DataFrame, mean_velocity: int, depot_node: int,
              depot: dict, node: int = None) -> tuple[int, int]:
    time = 0
    distance = 0

    position = truck.get_position()
    target_node = truck.get_cell().get_center_node() if node is None else node

    if truck.get_load() == 0:
        distance = distance_matrix.loc[truck.get_position(), depot_node]
        velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
        time += int(distance * 3.6 / velocity_kmh)
        if len(depot) > 15:
            bikes = {key: depot.pop(key) for key in list(depot.keys())[:15]}
            truck.set_load(bikes)
        position = depot_node

    if position != target_node:
        distance = distance_matrix.loc[position, target_node]
        velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
        time += int(distance * 3.6 / velocity_kmh)
        truck.set_position(target_node)

    truck.leaving_cell = truck.get_cell()

    return time, distance


def pick_up_bike(truck: Truck, station_dict: dict[int, Station], distance_matrix: pd.DataFrame,
                 mean_velocity: int, depot_node: int, depot: dict, system_bikes: dict) -> tuple[int, int, bool]:
    cell = truck.get_cell()
    bike_dict = {}
    for station_id in cell.get_nodes():
        bike_dict.update(station_dict[station_id].get_bikes())

    # Flag no bike picked up
    if cell.get_total_bikes() == 0:
        return 0, 0, False

    # Compute the metric for each bike
    max_distance = 0
    bikes_metric = {}
    for station_id in cell.get_nodes():
        if station_dict.get(station_id).get_number_of_bikes() > 0:
            distance = distance_matrix.loc[truck.get_position(), station_id]
            if distance > max_distance:
                max_distance = distance
            for bike_id, bike in station_dict.get(station_id).get_bikes().items():
                norm_batt = bike.get_battery() / bike.get_max_battery()
                if distance != 0:
                    bikes_metric[bike_id] = distance * norm_batt
                else:
                    bikes_metric[bike_id] = norm_batt

    # Normalize the metric
    if max_distance != 0:
        for bike_id in bikes_metric.keys():
            bikes_metric[bike_id] = bikes_metric[bike_id] / max_distance

    # Find the lowest metric bike
    bike_id = min(bikes_metric, key=bikes_metric.get)
    station = bike_dict[bike_id].get_station()

    distance = distance_matrix.loc[truck.get_position(), station.get_station_id()]
    velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
    time = int(distance * 3.6 / velocity_kmh)

    bike = station.unlock_bike(bike_id)
    if bike_id in system_bikes.keys():
        system_bikes.pop(bike_id)
    else:
        raise ValueError("Bike not in system")

    try:
        truck.load_bike(bike)
    except ValueError:
        distance = distance_matrix.loc[truck.get_position(), depot_node]
        velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
        t_reload = 2*int(distance * 3.6 / velocity_kmh)
        time += t_reload
        while truck.get_load() > 15:
            bk = truck.unload_bike()
            depot[bk.get_bike_id()] = bk
        truck.load_bike(bike)
    truck.set_position(station.get_station_id())

    truck.leaving_cell = truck.get_cell()

    return time, distance, True


def charge_bike(truck: Truck, station_dict: dict[int, Station], distance_matrix: pd.DataFrame,
                mean_velocity: int, depot_node: int, depot: dict, system_bikes: dict) -> tuple[int, int, bool]:
    return pick_up_bike(truck, station_dict, distance_matrix, mean_velocity, depot_node, depot, system_bikes)


def stay(truck: Truck) -> int:
    truck.leaving_cell = truck.get_cell()
    return 0
