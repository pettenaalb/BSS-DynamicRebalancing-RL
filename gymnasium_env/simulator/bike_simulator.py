import pandas as pd
import numpy as np

from gymnasium_env.simulator.station import Station
from gymnasium_env.simulator.bike import Bike
from gymnasium_env.simulator.trip import Trip
from gymnasium_env.simulator.event import EventType, Event
from gymnasium_env.simulator.utils import generate_poisson_events, truncated_gaussian, Logger

# ----------------------------------------------------------------------------------------------------------------------

def departure_handler(trip: Trip, station_dict: dict, nearby_nodes_dict: dict[str, dict[str, tuple]],
                      distance_matrix: pd.DataFrame) -> Trip:
    """
    Handle the departure of a trip from the starting station.

    Parameters:
        - trip (Trip): The trip object to be processed.
        - station_dict (dict): A dictionary containing the stations in the network.
        - nearby_nodes_dict (dict): A dictionary containing the nearby nodes for each station.
        - distance_matrix (pd.DataFrame): A DataFrame containing the distance matrix between stations.

    Returns:
        - Trip: The trip object after processing.
    """
    start_station = trip.get_start_location()
    start_station_id = start_station.get_station_id()

    # Check if there are any bikes available at the starting station
    if len(start_station.get_bikes()) > 0:
        bike = start_station.unlock_bike()
        if bike.get_battery() > trip.get_distance()/1000:
            trip.set_bike(bike)
            trip.set_failed(False)
            return trip
        else:
            start_station.lock_bike(bike)

    # Check if there are any bikes available at nearby stations
    nodes_dist_dict = {node_id: distance_matrix.at[start_station_id, node_id] for node_id in nearby_nodes_dict[start_station_id]}
    sorted_nodes = {k: v for k, v in sorted(nodes_dist_dict.items(), key=lambda item: item[1])}
    for node_id in sorted_nodes:
        if len(station_dict[node_id].get_bikes()) > 0:
            bike = station_dict[node_id].unlock_bike()
            if bike.get_battery() > trip.get_distance()/1000:
                trip.set_deviated_location(station_dict[node_id])
                trip.set_bike(bike)
                trip.set_failed(False)
                trip.set_deviated(True)
                return trip
            else:
                station_dict[node_id].lock_bike(bike)

    trip.set_failed(True)
    return trip


def arrival_handler(trip: Trip, system_bikes: dict[int, Bike], outside_system_bikes: dict[int, Bike]) -> tuple[dict[int, Bike], dict[int, Bike]]:
    """
    Handle the arrival of a trip at the destination station.

    Parameters:
        - trip (Trip): The trip object to be processed.
    """
    if trip.is_failed():
        return system_bikes, outside_system_bikes

    start_station = trip.get_start_location()
    end_station = trip.get_end_location()
    bike = trip.get_bike()
    end_station.lock_bike(bike)

    # Move the bike to the outside system if the destination station is outside the system
    if end_station.get_station_id() == 10000:
        outside_system_bikes[bike.get_bike_id()] = system_bikes.pop(bike.get_bike_id())

    # Move the bike back to the system if the starting station is outside the system
    if start_station.get_station_id() == 10000:
        system_bikes[bike.get_bike_id()] = outside_system_bikes.pop(bike.get_bike_id())

    return system_bikes, outside_system_bikes

# ----------------------------------------------------------------------------------------------------------------------

def simulate_environment(duration: int, time_slot: int, global_rate: float, pmf: pd.DataFrame, stations: dict,
                         distance_matrix: pd.DataFrame, residual_event_buffer: list = None) -> list[Event]:
    """
    Simulate the environment for a given time interval.

    Parameters:
        - time_interval (int): The time interval to simulate.
        - time_slot (int): The time slot to simulate.
        - rate (float): The rate of requests.
        - pmf (pd.DataFrame): The PMF matrix.
        - station_dict (dict): A dictionary containing the stations in the network.
        - distance_matrix (pd.DataFrame): A DataFrame containing the distance matrix between stations.

    Returns:
        - list[Event]: A list of events generated during the simulation.
    """
    # Simulate requests
    event_times = generate_poisson_events(global_rate, duration)
    event_buffer = []

    for event_time in event_times:
        random_value = np.random.rand()
        stn_pair = tuple(pmf.iloc[np.searchsorted(pmf['cumsum'].values, random_value)]['id'])
        if stn_pair[0] == 10000:
            ev_t = event_time + 3600*(3*time_slot + 1)
            trip = Trip(ev_t, ev_t, stations[stn_pair[0]], stations[stn_pair[1]])
            arr_event = Event(time=event_time, event_type=EventType.ARRIVAL, trip=trip)
            event_buffer.append(arr_event)
        else:
            velocity_kmh = truncated_gaussian(5, 25, 15, 5)
            if stn_pair[1] == 10000:
                # TODO: Implement a more realistic distance model for trips outside the system
                distance = truncated_gaussian(2, 7, 4.5, 1)
            else:
                distance = distance_matrix.at[stn_pair[0], stn_pair[1]]
            travel_time_seconds = int(distance * 3.6 / velocity_kmh)
            ev_t = event_time + 3600*(3*time_slot + 1)
            trip = Trip(ev_t, ev_t + travel_time_seconds, stations[stn_pair[0]],
                        stations[stn_pair[1]], distance=distance)
            dep_event = Event(time=event_time, event_type=EventType.DEPARTURE, trip=trip)
            arr_event = Event(time=event_time + travel_time_seconds, event_type=EventType.ARRIVAL, trip=trip)
            event_buffer.append(dep_event)
            event_buffer.append(arr_event)

    if residual_event_buffer:
        event_buffer.extend(residual_event_buffer)

    # Sort the event buffer based on time
    event_buffer.sort(key=lambda x: x.time)

    return event_buffer


def event_handler(event: Event, station_dict: dict[int, Station], nearby_nodes_dict: dict[str, dict[str, tuple]],
                  distance_matrix: pd.DataFrame, system_bikes: dict[int, Bike], outside_system_bikes: dict[int, Bike],
                  logger: Logger) -> tuple[int, dict[int, Bike], dict[int, Bike]]:
    """
    Handle the event based on its type.

    Parameters:
        - event (Event): The event object to be processed.
        - station_dict (dict): A dictionary containing the stations in the network.
        - nearby_nodes_dict (dict): A dictionary containing the nearby nodes for each station.
        - distance_matrix (pd.DataFrame): A DataFrame containing the distance matrix between stations.
        - system_bikes (dict): A dictionary containing the bikes in the system.
        - outside_system_bikes (dict): A dictionary containing the bikes outside the system.

    Returns:
        - bool: A boolean indicating whether the event failed or not.
        - dict: A dictionary containing the bikes in the system.
        - dict: A dictionary containing the bikes outside the system.
    """
    failure = 0
    if event.is_departure():
        trip = departure_handler(event.get_trip(), station_dict, nearby_nodes_dict, distance_matrix)
        if trip.is_failed():
            failure = 1
            logger.log_no_available_bikes(trip.get_start_location().get_station_id(), trip.get_end_location().get_station_id())
        else:
            logger.log_trip(trip)
    else:
        system_bikes, outside_system_bikes = arrival_handler(event.get_trip(), system_bikes, outside_system_bikes)

    return failure, system_bikes, outside_system_bikes