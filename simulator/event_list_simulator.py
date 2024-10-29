import os
import random

import numpy
import numpy as np
import pandas

import networkx as nx
import osmnx as ox
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

from simulator.utils import poisson_simulation
from station import Station
from bike import Bike
from trip import Trip
import utils

system_bikes = []
schedule_buffer = []
failures = 0
rate_matrix = pd.DataFrame()

def initialize_bikes(stn: Station):
    bikes = []

    n = random.randint(0, 2)

    for i in range(n):
        bikes.append(Bike(stn=stn))

    return bikes


def initialize_rate_matrix(gdf_nodes: pd.DataFrame) -> DataFrame:
    osm_ids = gdf_nodes.index.tolist()

    rates = np.zeros((len(osm_ids), len(osm_ids)))
    rate_df = pd.DataFrame(rates, index=osm_ids, columns=osm_ids)

    for i in osm_ids:
        for j in osm_ids:
            if i != j:
                rate_df.at[i, j] = 0.0001

    return rate_df


def initialize_stations():
    if os.path.isfile("../data/cambridge_network.graphml"):
        print("Loading the network data...")
        graph = ox.load_graphml("../data/cambridge_network.graphml")
    else:
        print("Downloading the network data...")
        place = "Cambridge, Massachusetts, USA"
        graph = ox.graph_from_place(place, network_type="drive")
        ox.save_graphml(graph, "../data/cambridge_network.graphml")

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)

    stations = []

    for index, row in gdf_nodes.iterrows():
        stations.append(Station(index, row["y"], row["x"]))
        bikes = initialize_bikes(stations[-1])
        global system_bikes
        system_bikes.extend(bikes)
        stations[-1].set_bikes(bikes)

    global rate_matrix
    rate_matrix = initialize_rate_matrix(gdf_nodes)

    return stations

def simulate_requests(time_interval: int):
    indices = rate_matrix.index.tolist()
    request_simulations = pd.DataFrame([[[] for _ in range(len(indices))] for _ in range(len(indices))],
                                       index=indices, columns=indices)

    for i in indices:
        for j in indices:
            rate = rate_matrix.at[i, j]
            if rate != 0:
                _, event_times, _ = poisson_simulation(rate, time_interval)
                request_simulations.at[i, j] = event_times
            else:
                request_simulations.at[i, j] = None

    return request_simulations

def schedule_request(start_time: int, end_time: int, start_location: Station, end_location: Station) -> bool:
    if len(start_location.get_bikes()) > 0:
        bike = start_location.unlock_bike()
        bike.set_availability(False)
        trip = Trip(start_time, end_time, start_location, end_location, bike)
        global schedule_buffer
        schedule_buffer.append(trip)
        print("Trip scheduled: ", trip)
        return True

    global failures
    failures += 1
    return False

def event_scheduler(time: int, stations: list, request_simulations: pd.DataFrame):
    # Check for requests
    for i, start_station in enumerate(stations):
        for j, end_station in enumerate(stations):
            event_times = request_simulations.at[start_station.get_station_id(), end_station.get_station_id()]
            if event_times is not None:
                if time in event_times:
                    # Schedule a trip if a request occurs
                    trip_duration = random.randint(5, 15) * 60  # Random duration between 5 to 15 minutes
                    schedule_request(time, time + trip_duration, start_station, end_station)

    # Drop finished trips
    global schedule_buffer
    for trip in schedule_buffer:
        if trip.get_end_time() <= time:
            bike = trip.get_bike()
            bike.set_availability(True)
            end_location = trip.get_end_location()
            end_location.lock_bike(bike)
            schedule_buffer.remove(trip)

def simulate_environment(time_interval: int, stations: list, request_simulations: pd.DataFrame):
    for t in range(time_interval):
        event_scheduler(t, stations, request_simulations)

def main():
    ox.settings.use_cache = True

    stations = initialize_stations()

    # request_simulations = simulate_requests(3600)

    # simulate_environment(3600, stations, request_simulations) # 1 hour


if __name__ == '__main__':
    main()
