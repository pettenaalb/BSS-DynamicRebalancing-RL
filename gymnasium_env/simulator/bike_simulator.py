import pandas as pd
import numpy as np

from gymnasium_env.simulator.station import Station
from gymnasium_env.simulator.bike import Bike
from gymnasium_env.simulator.trip import Trip
from gymnasium_env.simulator.event import EventType, Event
from gymnasium_env.simulator.utils import generate_poisson_events, truncated_gaussian, Logger, initialize_bikes

# ----------------------------------------------------------------------------------------------------------------------

def event_handler(event: Event, station_dict: dict[int, Station], nearby_nodes_dict: dict[str, dict[str, tuple]],
                  distance_matrix: pd.DataFrame, system_bikes: dict[int, Bike], outside_system_bikes: dict[int, Bike],
                  next_bike_id: int, logger: Logger) -> tuple[int, int]:
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
        trip, next_bike_id = departure_handler(event.get_trip(), station_dict, nearby_nodes_dict, distance_matrix,
                                 outside_system_bikes, next_bike_id)
        if trip.is_failed():
            failure = 1
            logger.log_no_available_bikes(trip.get_start_location().get_station_id(), trip.get_end_location().get_station_id())
        else:
            logger.log_trip(trip)
    else:
        arrival_handler(event.get_trip(), system_bikes, outside_system_bikes)

    return failure, next_bike_id


def departure_handler(trip: Trip, station_dict: dict, nearby_nodes_dict: dict[str, dict[str, tuple]],
                      distance_matrix: pd.DataFrame, outside_system_bikes: dict[int, Bike], next_bike_id: int) -> tuple[Trip, int]:
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

    # Check if the starting station is outside the system
    if start_station_id == 10000:
        if len(outside_system_bikes) == 0:
            bikes, next_bike_id = initialize_bikes(start_station, n=100, next_bike_id=next_bike_id)
            outside_system_bikes.update(bikes)
        bike = outside_system_bikes.pop(next(iter(outside_system_bikes)))
        trip.set_bike(bike)
        trip.set_failed(False)
        return trip, next_bike_id

    # Check if there are any bikes available at the starting station
    if start_station.get_number_of_bikes() > 0:
        bike = start_station.unlock_bike()
        if bike.get_battery() > trip.get_distance()/1000:
            trip.set_bike(bike)
            trip.set_failed(False)
            return trip, next_bike_id
        else:
            start_station.lock_bike(bike)

    # Check if there are any bikes available at nearby stations
    nodes_dist_dict = {node_id: distance_matrix.at[start_station_id, node_id] for node_id in nearby_nodes_dict[start_station_id]}
    for node_id, _ in sorted(nodes_dist_dict.items(), key=lambda item: item[1]):
        if station_dict[node_id].get_number_of_bikes() > 0:
            bike = station_dict[node_id].unlock_bike()
            if bike.get_battery() > trip.get_distance()/1000:
                trip.set_deviated_location(station_dict[node_id])
                trip.set_bike(bike)
                trip.set_failed(False)
                trip.set_deviated(True)
                return trip, next_bike_id
            else:
                station_dict[node_id].lock_bike(bike)

    trip.set_failed(True)
    return trip, next_bike_id


def arrival_handler(trip: Trip, system_bikes: dict[int, Bike], outside_system_bikes: dict[int, Bike]):
    """
    Handle the arrival of a trip at the destination station.

    Parameters:
        - trip (Trip): The trip object to be processed.
    """
    if trip.is_failed():
        return

    start_station = trip.get_start_location()
    end_station = trip.get_end_location()
    bike = trip.get_bike()

    # Move the bike to the outside system if the destination station is outside the system
    if end_station.get_station_id() == 10000:
        outside_system_bikes[bike.get_bike_id()] = system_bikes.pop(bike.get_bike_id())
        return

    # Move the bike back to the system if the starting station is outside the system
    if start_station.get_station_id() == 10000:
        system_bikes[bike.get_bike_id()] = outside_system_bikes.pop(bike.get_bike_id())

    end_station.lock_bike(bike)

# ----------------------------------------------------------------------------------------------------------------------

def simulate_environment(duration: int, timeslot: int, global_rate: float, pmf: pd.DataFrame, stations: dict,
                         distance_matrix: pd.DataFrame) -> list[Event]:
    """
    Simulate the environment for a given time interval.

    Parameters:
        - time_interval (int): The time interval to simulate.
        - timeslot (int): The time slot to simulate.
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
        if stn_pair[0] == 10000 or stn_pair[1] == 10000:
            distance = 0
            travel_time_seconds = 1
        else:
            velocity_kmh = truncated_gaussian(5, 25, 15, 5)
            distance = distance_matrix.at[stn_pair[0], stn_pair[1]]
            travel_time_seconds = int(distance * 3.6 / velocity_kmh)

        ev_t = event_time + 3600*(3*timeslot + 1)
        trip = Trip(ev_t, ev_t + travel_time_seconds, stations[stn_pair[0]],
                    stations[stn_pair[1]], distance=distance)

        dep_event = Event(time=event_time, event_type=EventType.DEPARTURE, trip=trip)
        arr_event = Event(time=event_time + travel_time_seconds, event_type=EventType.ARRIVAL, trip=trip)

        event_buffer.append(dep_event)
        event_buffer.append(arr_event)

    # Sort the event buffer based on time
    event_buffer.sort(key=lambda x: x.time)

    return event_buffer