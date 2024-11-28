import logging

import pandas as pd
import numpy as np

from tqdm import tqdm

from gymnasium_env.simulator.bike import Bike
from gymnasium_env.simulator.trip import Trip
from gymnasium_env.simulator.event import EventType, Event
from gymnasium_env.simulator.utils import generate_poisson_events, truncated_gaussian

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
                start_station.lock_bike(bike)

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

def simulate_environment(time_interval: int, time_slot: int, rate: float, pmf: pd.DataFrame, station_dict: dict,
                         nearby_nodes_dict: dict[str, dict[str, tuple]], distance_matrix: pd.DataFrame,
                         system_bikes: dict[int, Bike], outside_system_bikes: dict[int, Bike]) -> tuple[int, int, dict[int, Bike], dict[int, Bike]]:
    """
    Simulate the environment for a given time interval.

    Parameters:
        - time_interval (int): The time interval for the simulation.
        - stations (list): A list of Station objects.
        - request_simulations (pd.DataFrame): A DataFrame containing the simulated request times for each station pair.
    """
    # Simulate requests
    event_times = generate_poisson_events(rate, time_interval)
    event_buffer = []
    tbar = tqdm(total=len(event_times), desc="Setting events", position=0, leave=True)
    for event_time in event_times:
        random_value = np.random.rand()
        stn_pair = tuple(pmf.iloc[np.searchsorted(pmf['cumsum'].values, random_value)]['id'])
        if stn_pair[0] == 10000:
            ev_t = event_time + 3600*(time_slot + 1)
            trip = Trip(ev_t, ev_t, station_dict[stn_pair[0]], station_dict[stn_pair[1]])
            arr_event = Event(time=event_time, event_type=EventType.ARRIVAL, trip=trip)
            event_buffer.append(arr_event)
        else:
            velocity_kmh = truncated_gaussian(5, 25, 15, 5)
            if stn_pair[1] == 10000:
                distance = truncated_gaussian(2, 7, 4.5, 1)
            else:
                distance = distance_matrix.at[stn_pair[0], stn_pair[1]]
            travel_time_seconds = int(distance * 3.6 / velocity_kmh)
            ev_t = event_time + 3600*(3*time_slot + 1)
            trip = Trip(ev_t, ev_t + travel_time_seconds, station_dict[stn_pair[0]],
                        station_dict[stn_pair[1]], distance=distance)
            dep_event = Event(time=event_time, event_type=EventType.DEPARTURE, trip=trip)
            arr_event = Event(time=event_time + travel_time_seconds, event_type=EventType.ARRIVAL, trip=trip)
            event_buffer.append(dep_event)
            event_buffer.append(arr_event)

        tbar.update(1)

    # Sort the event buffer based on time
    event_buffer.sort(key=lambda x: x.time)
    failures = 0
    total_trips = 0

    tbar = tqdm(total=len(event_buffer), desc="Processing events", position=0, leave=True)

    for event in event_buffer:
        if event.is_departure():
            trip = departure_handler(event.get_trip(), station_dict, nearby_nodes_dict, distance_matrix)
            if trip.is_failed():
                failures += 1
                logging.warning(f"No bike available from station {trip.get_start_location().get_station_id()} to station {trip.get_end_location().get_station_id()}")
            else:
                total_trips += 1
                logging.info("Trip scheduled: %s", trip)
        else:
            system_bikes, outside_system_bikes = arrival_handler(event.get_trip(), system_bikes, outside_system_bikes)

        tbar.update(1)

    return total_trips, failures, system_bikes, outside_system_bikes
