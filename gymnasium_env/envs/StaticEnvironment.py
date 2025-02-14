import math
import pickle
import random
import bisect

import gymnasium as gym
import numpy as np
import pandas as pd
import osmnx as ox

from gymnasium.utils import seeding

from gymnasium_env.simulator.bike_simulator import simulate_environment, event_handler
from gymnasium_env.simulator.utils import initialize_graph, initialize_stations, initialize_bikes, truncated_gaussian
from gymnasium_env.simulator.truck_simulator import tsp_rebalancing

params = {
    'graph_file': 'utils/cambridge_network.graphml',
    'cell_file': 'utils/cell_data.pkl',
    'distance_matrix_file': 'utils/distance_matrix.csv',
    'filtered_stations_file': 'utils/filtered_stations.csv',
    'matrices_folder': 'matrices/09-10',
    'rates_folder': 'rates/09-10',
    'trips_folder': 'trips/',
    'nearby_nodes': 'utils/nearby_nodes.pkl',
    'velocity_matrix': 'utils/ev_velocity_matrix.csv',
    'consumption_matrix': 'utils/ev_consumption_matrix.csv'
}

days2num = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
num2days = {0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday', 4: 'friday', 5: 'saturday', 6: 'sunday'}

# ----------------------------------------------------------------------------------------------------------------------

class StaticEnv(gym.Env):

    # Initialize the environment
    def __init__(self, data_path: str):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(0,), dtype=np.float32)

        # Initialize paths and logger
        self.data_path = data_path

        # Initialize the graph
        self.graph = initialize_graph(data_path + params['graph_file'])

        # Compute the bounding box
        nodes, _ = ox.graph_to_gdfs(self.graph)
        self.min_lat, self.max_lat = nodes['y'].min(), nodes['y'].max()
        self.min_lon, self.max_lon = nodes['x'].min(), nodes['x'].max()

        # Initialize the nodes dictionary
        nodes_gdf = ox.graph_to_gdfs(self.graph, edges=False)
        self.nodes_dict = {node_id: (row['y'], row['x']) for node_id, row in nodes_gdf.iterrows()}

        # Load nearby nodes dictionary
        with open(data_path + params['nearby_nodes'], 'rb') as file:
            self.nearby_nodes_dict = pickle.load(file)

        # Initialize the cells
        with open(data_path + params['cell_file'], 'rb') as file:
            self.cells = pickle.load(file)

        # Initialize the distance matrix
        self.distance_matrix = pd.read_csv(data_path + params['distance_matrix_file'], index_col='osmid')
        self.distance_matrix.index = self.distance_matrix.index.astype(int)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(int)

        # Initialize the velocity matrix
        self.velocity_matrix = pd.read_csv(data_path + params['velocity_matrix'], index_col='hour')

        # Initialize the consumption matrix
        self.consumption_matrix = pd.read_csv(data_path + params['consumption_matrix'], index_col='hour')

        # Initialize simulation state variables
        self.pmf_matrix, self.global_rate, self.global_rate_dict = None, None, None
        self.system_bikes, self.outside_system_bikes, self.maximum_number_of_bikes = None, None, 0
        self.env_time, self.timeslot, self.day = 0, 0, 'monday'
        self.timeslots_completed, self.days_completed, self.total_timeslots = 0, 0, 0
        self.depot, self.depot_node = None, None
        self.stations = None
        self.event_buffer =[]
        self.cell_subgraph = None
        self.next_bike_id = 0

        self.num_rebalancing_events, self.rebalancing_hours = 0, []


    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        # Call parent class reset
        super().reset(seed=seed)

        # Day and time slot options
        self.day = options.get('day', 'monday') if options else 'monday'
        self.timeslot = options.get('timeslot', 0) if options else 0
        self.total_timeslots = options.get('total_timeslots', 56) if options else 56
        self.num_rebalancing_events = options.get('num_rebalancing_events', 1) if options else 1

        # Bike options
        if options:
            self.maximum_number_of_bikes = options.get('maximum_number_of_bikes', self.maximum_number_of_bikes)

        # Set rebalancing hours
        if self.num_rebalancing_events > 0:
            self.rebalancing_hours = [i+3 for i in range(0, 24, 24 // self.num_rebalancing_events)]
        self.timeslots_completed = 0
        self.days_completed = 0
        self.event_buffer = []

        # Reset the cells
        for cell in self.cells.values():
            cell.reset()

        # Initialize the depot
        self.next_bike_id = 0
        self.depot_node = self.cells.get(options.get('depot_id', 491) if options else 491).get_center_node()
        self.depot, self.next_bike_id = initialize_bikes(n=self.maximum_number_of_bikes, next_bike_id=self.next_bike_id)

        # Create stations dictionary
        from gymnasium_env.simulator.station import Station
        gdf_nodes = ox.graph_to_gdfs(self.graph, edges=False)
        stns = {}
        for index, row in gdf_nodes.iterrows():
            station = Station(index, row['y'], row['x'])
            stns[index] = station
        stns[10000] = Station(10000, 0, 0)
        self.stations = stns

        # Set the cell for each station
        for cell in self.cells.values():
            for node in cell.nodes:
                self.stations[node].set_cell(cell)

        for station in self.stations.values():
            if station.get_cell() is None and station.get_station_id() != 10000:
                raise ValueError(f"Station {station} is not assigned to a cell.")

        # Load the PMF matrix and global rate for the current day and time slot
        pmf_matrix, _ = self._load_pmf_matrix_global_rate(self.day, self.timeslot)

        # Initialize stations and system bikes
        bikes_per_station = {}
        std_dev = 0.0
        for stn_id, stn in self.stations.items():
            if stn_id != 10000:
                base_bikes = math.ceil(pmf_matrix.loc[stn_id, :].sum() * int(self.maximum_number_of_bikes))
                noise = random.gauss(0, std_dev) * base_bikes
                noisy_bikes = max(0, int(base_bikes + noise))
                noisy_bikes = min(noisy_bikes, stn.get_capacity())
                bikes_per_station[stn_id] = noisy_bikes

        # Adjust the total bikes to not exceed the desired total
        current_total = sum(bikes_per_station.values())
        while current_total > self.maximum_number_of_bikes:
            station_id = random.choice(list(bikes_per_station.keys()))
            if bikes_per_station[station_id] > 0:
                bikes_per_station[station_id] -= 1
                current_total -= 1

        # Initialize the system bikes
        self.system_bikes, self.outside_system_bikes, self.next_bike_id = initialize_stations(
            stations=self.stations,
            depot=self.depot,
            bikes_per_station=bikes_per_station,
            next_bike_id=self.next_bike_id,
        )

        # Initialize the day and time slot
        self._initialize_day()

        return {}, {}


    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:
        terminated = False
        total_failures = 0
        failures_per_timeslot = []
        rebalance_time = []
        while not terminated:
            self.env_time += 30

            # Process all events that occurred before the updated environment time
            while self.event_buffer and self.event_buffer[0].time < self.env_time:
                # print("Entering in while")
                event = self.event_buffer.pop(0)
                failure, self.next_bike_id = event_handler(
                    event=event,
                    station_dict=self.stations,
                    nearby_nodes_dict=self.nearby_nodes_dict,
                    distance_matrix=self.distance_matrix,
                    system_bikes=self.system_bikes,
                    outside_system_bikes=self.outside_system_bikes,
                    next_bike_id=self.next_bike_id
                )
                total_failures += failure

            if self.env_time % 3600 == 0:
                hour = (self.env_time // 3600) + 1
                if hour in self.rebalancing_hours:
                    time_to_rebalance = self._rebalance_system()
                    rebalance_time.append(time_to_rebalance)

            if self.env_time >= 3600*3*(self.timeslot + 1):
                self.timeslot = (self.timeslot + 1) % 8
                failures_per_timeslot.append(total_failures)
                total_failures = 0
                if self.timeslot == 0:
                    self.day = num2days[(days2num[self.day] + 1) % 7]
                    self.days_completed += 1
                    self._initialize_day()
                    terminated = True
                self.timeslots_completed += 1

        info = {
            'failures': failures_per_timeslot,
            'time': self.env_time + (self.timeslot * 3 + 1) * 3600,
            'day': self.day,
            'week': int(self.days_completed // 7),
            'rebalance_time': rebalance_time
        }

        done = True if self.timeslots_completed == self.total_timeslots else False

        return {}, 0, done, terminated, info


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _initialize_day(self):
        # Load PMF matrix and global rate for the current day and time slot
        total_events = []
        for timeslot in range(0, 8):
            pmf_matrix, global_rate = self._load_pmf_matrix_global_rate(self.day, timeslot)
            # Flatten the PMF matrix for event simulation
            values = pmf_matrix.values.flatten()
            ids = [(row, col) for row in pmf_matrix.index for col in pmf_matrix.columns]
            flattened_pmf = pd.DataFrame({'id': ids, 'value': values})
            flattened_pmf['cumsum'] = np.cumsum(flattened_pmf['value'].values)
            events = simulate_environment(
                duration=3600 * 3,  # 3 hours
                timeslot=timeslot,
                global_rate=global_rate,
                pmf=flattened_pmf,
                stations=self.stations,
                distance_matrix=self.distance_matrix,
            )
            for event in events:
                event.time += 3600 * 3 * timeslot
            total_events.extend(events)

        for event in total_events:
            bisect.insort(self.event_buffer, event, key=lambda x: x.time)

        # Initialize environment time
        self.env_time = 0


    def _load_pmf_matrix_global_rate(self, day: str, timeslot: int) -> tuple[pd.DataFrame, float]:
        # Load the PMF matrix and global rate for a given day and time slot
        matrix_path = self.data_path + params['matrices_folder'] + '/' + str(timeslot).zfill(2) + '/'
        pmf_matrix = pd.read_csv(matrix_path + day.lower() + '-pmf-matrix.csv', index_col='osmid')

        # Convert index and columns to integers
        pmf_matrix.index = pmf_matrix.index.astype(int)
        pmf_matrix.columns = pmf_matrix.columns.astype(int)
        pmf_matrix.loc[10000, 10000] = 0.0

        if self.global_rate_dict is None:
            with open(self.data_path + 'utils/global_rates.pkl', 'rb') as f:
                self.global_rate_dict = pickle.load(f)

        global_rate = self.global_rate_dict[(day.lower(), timeslot)]

        return pmf_matrix, global_rate


    def _rebalance_system(self) -> int:
        # Add bikes back to the system
        while len(self.system_bikes) < self.maximum_number_of_bikes:
            bike = self.outside_system_bikes.pop(iter(next(self.outside_system_bikes)))
            self.system_bikes[bike.get_bike_id()] = bike

        # Compute the net flow per cell
        net_flow_per_cell = {cell_id: 0 for cell_id in self.cells.keys()}
        for event in self.event_buffer:
            if event.is_departure():
                station_id = event.trip.get_start_location().get_station_id()
                if station_id != 10000:
                    cell = self.stations[station_id].get_cell()
                    net_flow_per_cell[cell.get_id()] -= 1
            elif event.is_arrival():
                station_id = event.trip.get_end_location().get_station_id()
                if station_id != 10000:
                    cell = self.stations[station_id].get_cell()
                    net_flow_per_cell[cell.get_id()] += 1

        # Assign bikes to cells based on the net flow
        bikes_per_cell = {cell_id: 5 for cell_id in self.cells.keys()}
        available_bikes = sum([1 for bike in self.system_bikes.values() if bike.available])
        left_bikes = available_bikes - 5*len(self.cells)
        total_negative_flow = sum(flow for flow in net_flow_per_cell.values() if flow < 0)
        used_bikes = 0
        for cell_id, flow in net_flow_per_cell.items():
            if flow < 0:
                num_of_bikes = int((flow / total_negative_flow) * left_bikes)
                bikes_per_cell[cell_id] += num_of_bikes
                used_bikes += num_of_bikes

        # Assign the remaining bikes to cells with negative flow randomly
        if used_bikes < left_bikes:
            cell_ids = [cell_key for cell_key, flow in net_flow_per_cell.items() if flow < 0]
            random.shuffle(cell_ids)
            for cell_id in cell_ids:
                bikes_per_cell[cell_id] += 1
                used_bikes += 1
                if used_bikes == left_bikes:
                    break

        # Compute rebalance time
        surplus_nodes = {}
        deficit_nodes = {}
        for cell_id, cell in self.cells.items():
            cell_bikes = cell.get_total_bikes()
            target_bikes = bikes_per_cell[cell_id]
            if cell_bikes > target_bikes:
                surplus_nodes[cell.get_center_node()] = cell_bikes - target_bikes
            elif cell_bikes < target_bikes:
                deficit_nodes[cell.get_center_node()] = target_bikes - cell_bikes

        distance, _ = tsp_rebalancing(surplus_nodes, deficit_nodes, self.depot_node, self.distance_matrix)
        hour = (self.env_time // 3600) + 1
        mean_truck_velocity = self.velocity_matrix.loc[hour, self.day]
        velocity_kmh = truncated_gaussian(10, 70, mean_truck_velocity, 5)
        time = int(distance * 3.6 / velocity_kmh)

        # Empty the stations
        for station in self.stations.values():
            if station.get_station_id() != 10000:
                while station.get_number_of_bikes() > 0:
                    bike = station.unlock_bike()
                    bike.set_availability(True)

        # Available bikes dictionary
        available_bikes = {bike_id: bike for bike_id, bike in self.system_bikes.items() if bike.available}
        for cell_id, num_of_bikes in bikes_per_cell.items():
            for _ in range(num_of_bikes):
                bike = available_bikes.pop(next(iter(available_bikes)))
                center_node_id = self.cells[cell_id].get_center_node()
                self.stations[center_node_id].lock_bike(bike)

        # Charge all bikes
        for bike in self.system_bikes.values():
            bike.set_battery(bike.get_max_battery())

        return time