import networkx as nx
import pandas as pd

from networkx.algorithms.approximation import traveling_salesman_problem

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
            bk.reset()
            depot[bk.get_bike_id()] = bk
        truck.load_bike(bike)
    truck.set_position(station.get_station_id())

    truck.leaving_cell = truck.get_cell()

    return time, distance, True


def charge_bike(truck: Truck, station_dict: dict[int, Station], distance_matrix: pd.DataFrame,
                mean_velocity: int, depot_node: int, depot: dict, system_bikes: dict) -> tuple[int, int, bool]:
    cell = truck.get_cell()
    bike_dict = {}
    for station_id in cell.get_nodes():
        bike_dict.update(station_dict[station_id].get_bikes())

    # Flag no bike picked up
    if cell.get_total_bikes() == 0:
        return 0, 0, False

    # Find the lowest metric bike
    bike_charge = {bike_id: bike.get_battery() for bike_id, bike in bike_dict.items()}
    bike_id = min(bike_charge, key=bike_charge.get)
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
        t_reload = 2 * int(distance * 3.6 / velocity_kmh)
        time += t_reload
        while truck.get_load() > 15:
            bk = truck.unload_bike()
            bk.reset()
            depot[bk.get_bike_id()] = bk
        truck.load_bike(bike)
    truck.set_position(station.get_station_id())

    truck.leaving_cell = truck.get_cell()

    # return pick_up_bike(truck, station_dict, distance_matrix, mean_velocity, depot_node, depot, system_bikes)
    return time, distance, True


def stay(truck: Truck) -> int:
    truck.leaving_cell = truck.get_cell()
    return 0

# ----------------------------------------------------------------------------------------------------------------------

def tsp_rebalancing(surplus_nodes: dict, deficit_nodes: dict, starting_node, distance_matrix: pd.DataFrame):
    all_nodes = list(surplus_nodes.keys()) + list(deficit_nodes.keys())
    tsp_graph = nx.Graph()

    # Check if there are nodes to process
    if not all_nodes:
        raise ValueError("No valid surplus or deficit nodes to rebalance.")

    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            node_i, node_j = all_nodes[i], all_nodes[j]
            distance = distance_matrix.loc[node_i, node_j]
            tsp_graph.add_edge(node_i, node_j, weight=distance)

    # Ensure starting node is included
    for node in all_nodes:
        tsp_graph.add_edge(starting_node, node, weight=distance_matrix.loc[starting_node, node])

    # Solve TSP to get the initial path
    tsp_path = traveling_salesman_problem(tsp_graph, cycle=False)

    # Variables to track progress
    total_distance = 0
    truck_bikes = 0
    final_route = []
    skipped_deficit_nodes = {}
    total_missing_bikes = 0

    # Process the TSP path dynamically
    current_node = starting_node

    for node in tsp_path:
        if node not in surplus_nodes and node not in deficit_nodes:
            continue

        distance = distance_matrix.loc[current_node, node]
        total_distance += distance
        final_route.append(node)

        # If it's a surplus node, pick up bikes, otherwise drop them
        if node in surplus_nodes:
            truck_bikes += surplus_nodes.pop(node, 0)
        elif node in deficit_nodes:
            deficit_demand = -deficit_nodes[node]
            if truck_bikes >= deficit_demand:
                # Enough bikes: drop them
                truck_bikes -= deficit_demand
                deficit_nodes.pop(node)
            else:
                # Not enough bikes: track it and move on
                skipped_deficit_nodes[node] = deficit_demand
                total_missing_bikes += deficit_demand

        # Move to the next node
        current_node = node

        # Once the truck has enough bikes to satisfy all skipped deficits, go back in an efficient order
        if 0 < total_missing_bikes <= truck_bikes:
            # Solve a new TSP for skipped deficit nodes
            backtrack_graph = nx.Graph()
            skipped_list = list(skipped_deficit_nodes.keys())

            for i in range(len(skipped_list)):
                for j in range(i + 1, len(skipped_list)):
                    node_i, node_j = skipped_list[i], skipped_list[j]
                    distance = distance_matrix.loc[node_i, node_j]
                    backtrack_graph.add_edge(node_i, node_j, weight=distance)

            # Ensure we start from the last node we visited
            for n in skipped_list:
                backtrack_graph.add_edge(current_node, n, weight=distance_matrix.loc[current_node, n])

            # Solve TSP for revisiting skipped nodes
            backtrack_path = traveling_salesman_problem(backtrack_graph, cycle=False)

            # Visit skipped nodes
            for n in backtrack_path:
                if n in skipped_deficit_nodes:
                    distance = distance_matrix.loc[current_node, n]
                    total_distance += distance
                    final_route.append(n)

                    # Drop bikes
                    bikes_needed = skipped_deficit_nodes[n]
                    truck_bikes -= bikes_needed
                    total_missing_bikes -= bikes_needed
                    skipped_deficit_nodes.pop(n)

                    # Move to the next node
                    current_node = n

    return total_distance, final_route