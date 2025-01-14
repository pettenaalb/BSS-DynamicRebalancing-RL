import math
import pickle
import threading
import random
import bisect

import gymnasium as gym
import numpy as np
import pandas as pd
import osmnx as ox

from gymnasium import spaces
from gymnasium.utils import seeding

from gymnasium_env.simulator.bike_simulator import simulate_environment, event_handler
from gymnasium_env.simulator.truck_simulator import (move_up, move_down, move_left, move_right, drop_bike, pick_up_bike,
                                                     charge_bike, stay)
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.utils import (initialize_graph, initialize_stations, Logger, Actions, initialize_bikes,
                                           convert_seconds_to_hours_minutes, initialize_cells_subgraph,
                                           logistic_penalty_function)

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

class BostonCity(gym.Env):

    # Initialize the environment
    def __init__(self, data_path: str):
        super().__init__()

        # Initialize paths and logger
        self.data_path = data_path
        self.logger = Logger(data_path + 'env_output.log')

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
        self.depot_node = self.cells.get(491).get_center_node()

        # Initialize the distance matrix
        self.distance_matrix = pd.read_csv(data_path + params['distance_matrix_file'], index_col='osmid')
        self.distance_matrix.index = self.distance_matrix.index.astype(int)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(int)

        # Initialize the velocity matrix
        self.velocity_matrix = pd.read_csv(data_path + params['velocity_matrix'], index_col='hour')

        # Initialize the consumption matrix
        self.consumption_matrix = pd.read_csv(data_path + params['consumption_matrix'], index_col='hour')

        # Define action space
        self.action_space = spaces.Discrete(len(Actions))

        # Define observation space
        # Features: [1 agent features + 7 days + 24 hours]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1 + 7 + 24,),
            dtype=np.float32
        )

        # Initialize simulation state variables
        self.pmf_matrix, self.global_rate, self.global_rate_dict = None, None, None
        self.system_bikes, self.outside_system_bikes = None, None
        self.maximum_number_of_bikes = 3500
        self.current_cell_id = None
        self.stations = None
        self.truck = None
        self.event_buffer = None
        self.next_event_buffer = None
        self.env_time = 0
        self.energy_cost_per_time = 0
        self.timeslot = 0
        self.day = 'monday'
        self.cell_subgraph = None
        self.timeslots_completed = 0
        self.days_completed = 0
        self.next_bike_id = 0
        self.total_timeslots = 0
        self.background_thread = None
        self.discount_factor = 0.99

        self.depot, self.next_bike_id = initialize_bikes(n=self.maximum_number_of_bikes, next_bike_id=self.next_bike_id)

        # Set logging options
        self.logging = True
        self.logger.set_logging(self.logging)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        # Call parent class reset
        super().reset(seed=seed)

        # Update day and time slot if provided in options
        self.day = options.get('day', 'monday') if options else 'monday'
        self.timeslot = options.get('timeslot', 0) if options else 0
        self.discount_factor = options.get('discount_factor', 0.99) if options else 0.99

        self.total_timeslots = options.get('total_timeslots', 2*365*8) if options else 2*365*8
        self.maximum_number_of_bikes = options.get('maximum_number_of_bikes', self.maximum_number_of_bikes) if options else self.maximum_number_of_bikes

        # Create stations dictionary
        from gymnasium_env.simulator.station import Station
        gdf_nodes = ox.graph_to_gdfs(self.graph, edges=False)
        stations = {}
        for index, row in gdf_nodes.iterrows():
            station = Station(index, row['y'], row['x'])
            stations[index] = station
        stations[10000] = Station(10000, 0, 0)
        self.stations = stations

        # Initialize the truck
        self.current_cell_id = options.get('initial_cell', 185) if options else 185
        cell = self.cells[self.current_cell_id]
        max_truck_load = options.get('max_truck_load', 30) if options else 30

        # Pop out 15 bikes from the depot and load them into the truck
        bikes = {key: self.depot.pop(key) for key in list(self.depot.keys())[:15]}
        self.truck = Truck(cell.center_node, cell, bikes=bikes, max_load=max_truck_load)

        # Initialize the day and time slot
        self._initialize_day_timeslot(handle_first_events=True)

        # Initialize stations and system bikes
        bikes_per_station = {}
        std_dev = 1.0
        for stn_id in self.stations.keys():
            base_bikes = int(self.pmf_matrix.loc[stn_id, :].sum() * int(self.maximum_number_of_bikes*0.8))
            noise = random.gauss(0, std_dev) * base_bikes
            noisy_bikes = max(0, int(base_bikes + noise))
            bikes_per_station[stn_id] = noisy_bikes

        # Adjust the total bikes to not exceed the desired total
        current_total = sum(bikes_per_station.values())
        if current_total > self.maximum_number_of_bikes*0.8:
            # Scale down proportionally if we exceed the total
            scale_factor = self.maximum_number_of_bikes*0.8 / current_total
            bikes_per_station = {stn_id: int(bikes * scale_factor) for stn_id, bikes in bikes_per_station.items()}

        print(f"Current total bikes: {current_total}")

        # Initialize the system bikes
        self.system_bikes, self.outside_system_bikes, self.next_bike_id = initialize_stations(
            stations=self.stations,
            depot=self.depot,
            bikes_per_station=bikes_per_station,
            next_bike_id=self.next_bike_id,
        )

        # Set the number of bikes in each cell
        for cell in self.cells.values():
            for node in cell.nodes:
                self.stations[node].set_cell(cell)
                cell.set_total_bikes(cell.get_total_bikes() + self.stations[node].get_number_of_bikes())

        # Initialize the cell subgraph
        self.cell_subgraph = initialize_cells_subgraph(self.cells, self.nodes_dict, self.distance_matrix)

        # Update the graph with regional metrics
        self._update_graph()

        # Return the initial observation and an optional info dictionary
        observation = self._get_obs()
        info = {
            'cells_subgraph': self.cell_subgraph,
            'network_graph': self.graph,
            'cell_dict': self.cells,
            'nodes_dict': self.nodes_dict,
            'agent_position': self._get_truck_position(),
            'steps': 0,
            'number_of_system_bikes': len(self.system_bikes),
        }

        return observation, info


    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:
        # Log the start of the step
        self.logger.new_log_line()

        # Check for discrepancies in the depot + system bikes between total bikes
        self._adjust_depot_system_discrepancy()

        # Perform the action and log it
        t = 0
        distance = 0
        prev_position = self.truck.get_position()
        hours, _ = divmod((self.timeslot * 3 + 1) * 3600 + self.env_time, 3600)
        hours = hours % 24
        mean_velocity = self.velocity_matrix.loc[hours, self.day]
        bike_picked_up = False
        if action == Actions.STAY.value:
            t = stay()
            self.logger.log_starting_action('STAY', t)
        elif action == Actions.RIGHT.value:
            t, distance = move_right(self.truck, self.distance_matrix, self.cells, mean_velocity)
            self.logger.log_starting_action('RIGHT', t)
        elif action == Actions.UP.value:
            t, distance = move_up(self.truck, self.distance_matrix, self.cells, mean_velocity)
            self.logger.log_starting_action('UP', t)
        elif action == Actions.LEFT.value:
            t, distance = move_left(self.truck, self.distance_matrix, self.cells, mean_velocity)
            self.logger.log_starting_action('LEFT', t)
        elif action == Actions.DOWN.value:
            t, distance = move_down(self.truck, self.distance_matrix, self.cells, mean_velocity)
            self.logger.log_starting_action('DOWN', t)
        elif action == Actions.DROP_BIKE.value:
            t, distance = drop_bike(self.truck, self.distance_matrix, mean_velocity, self.depot_node, self.depot)
            self.logger.log_starting_action('DROP_BIKE', t)
        elif action == Actions.PICK_UP_BIKE.value:
            t, distance, _ = pick_up_bike(self.truck, self.stations, self.distance_matrix, mean_velocity, self.depot_node,
                                       self.depot, self.system_bikes)
            self.logger.log_starting_action('PICK_UP_BIKE', t)
        elif action == Actions.CHARGE_BIKE.value:
            t, distance, bike_picked_up = charge_bike(self.truck, self.stations, self.distance_matrix, mean_velocity,
                                                      self.depot_node, self.depot, self.system_bikes)
            self.logger.log_starting_action('CHARGE_BIKE', t)

        # Calculate steps and log the state
        steps = math.ceil(t / 30)
        self.logger.log_state(
            step=int(self.env_time / 30),
            time=convert_seconds_to_hours_minutes((self.timeslot * 3 + 1) * 3600 + self.env_time)
        )

        # Check if the environment time exceeds the maximum time
        remaining_steps = 0
        if self.env_time + steps*30 > 3600*3:
            steps = math.ceil((3600*3 - self.env_time) / 30)
            remaining_steps = math.ceil((t - steps*30) / 30)

        # Update the environment state
        failures = self._jump_to_next_state(steps)

        terminated = False
        if self.env_time >= 3600*3:
            residual_event_buffer = self.event_buffer
            for event in residual_event_buffer:
                event.time -= 3600*3
            if self.timeslot == 7:
                self.timeslot = 0
                self.day = num2days[(days2num[self.day] + 1) % 7]
                self.days_completed += 1
            else:
                self.timeslot += 1
            self._initialize_day_timeslot(residual_event_buffer)
            self.timeslots_completed += 1
            terminated = True

        if remaining_steps > 0:
            failures.extend(self._jump_to_next_state(remaining_steps))

        # Handle specific actions post-environment update
        if action in {Actions.DROP_BIKE.value, Actions.CHARGE_BIKE.value}:
            if bike_picked_up or action == Actions.DROP_BIKE.value:
                station = self.stations.get(self.truck.get_position())
                bike = self.truck.unload_bike()
                station.lock_bike(bike)
                self.system_bikes[bike.get_bike_id()] = bike

        # Log the ending action
        self.logger.log_ending_action(
            time=convert_seconds_to_hours_minutes((self.timeslot * 3 + 1) * 3600 + self.env_time)
        )

        # Perform a final state update after the last action
        # print("Entering in last state update")
        failures.extend(self._jump_to_next_state(steps=1))

        # Log truck state
        self.logger.log_truck(self.truck)

        # Compute the outputs
        observation = self._get_obs()
        reward = self._get_reward(steps+remaining_steps, failures, distance)
        self._update_graph()
        path = (prev_position, self.truck.get_position())

        info = {
            'cells_subgraph': self.cell_subgraph,
            'agent_position': self._get_truck_position(),
            'time': self.env_time + (self.timeslot * 3 + 1) * 3600,
            'day': self.day,
            'week': int(self.days_completed // 7),
            'year': int(self.days_completed // 365),
            'failures': failures,
            'path': path,
            'number_of_system_bikes': len(self.system_bikes),
            'steps': steps+remaining_steps,
        }

        if self.timeslots_completed == self.total_timeslots:   # 2 years
            done = True
            # Print the total number of bikes in the environment
            print(f"Total number of bikes in the environment: {len(self.system_bikes)}")
        else:
            done = False

        # Return the step results
        return observation, reward, done, terminated, info


    def _precompute_poisson_events(self):
        """Background thread for precomputing Poisson events."""
        timeslot = (self.timeslot + 1) % 8
        day = self.day
        if timeslot == 0:
            day = num2days[(days2num[self.day] + 1) % 7]

        # Flatten the PMF matrix for event simulation
        pmf_matrix, global_rate = self._load_pmf_matrix_global_rate(day, timeslot)

        # Flatten the PMF matrix for event simulation
        values = pmf_matrix.values.flatten()
        ids = [(row, col) for row in pmf_matrix.index for col in pmf_matrix.columns]
        flattened_pmf = pd.DataFrame({'id': ids, 'value': values})
        flattened_pmf['cumsum'] = np.cumsum(flattened_pmf['value'].values)

        self.next_event_buffer = simulate_environment(
            duration=3600 * 3,  # 3 hours
            timeslot=timeslot,
            global_rate=global_rate,
            pmf=flattened_pmf,
            stations=self.stations,
            distance_matrix=self.distance_matrix,
        )


    def _initialize_day_timeslot(self, residual_event_buffer: list = None, handle_first_events = False) -> int:
        # Load PMF matrix and global rate for the current day and time slot
        self.pmf_matrix, self.global_rate = self._load_pmf_matrix_global_rate(self.day, self.timeslot)

        for stn_id, stn in self.stations.items():
            stn.set_request_rate(self.pmf_matrix.loc[stn_id, :].sum()*self.global_rate)

        # Simulate the environment for the time slot
        if self.next_event_buffer is not None:
            self.event_buffer = self.next_event_buffer
            self.next_event_buffer = None
            if residual_event_buffer is not None:
                for event in residual_event_buffer:
                    bisect.insort(self.event_buffer, event, key=lambda x: x.time)
        else:
            # Flatten the PMF matrix for event simulation
            values = self.pmf_matrix.values.flatten()
            ids = [(row, col) for row in self.pmf_matrix.index for col in self.pmf_matrix.columns]
            flattened_pmf = pd.DataFrame({'id': ids, 'value': values})
            flattened_pmf['cumsum'] = np.cumsum(flattened_pmf['value'].values)
            self.event_buffer = simulate_environment(
                duration=3600 * 3,  # 3 hours
                timeslot=self.timeslot,
                global_rate=self.global_rate,
                pmf=flattened_pmf,
                stations=self.stations,
                distance_matrix=self.distance_matrix,
            )
            if residual_event_buffer is not None:
                for event in residual_event_buffer:
                    bisect.insort(self.event_buffer, event, key=lambda x: x.time)

        self.background_thread = threading.Thread(target=self._precompute_poisson_events)
        self.background_thread.start()

        # Initialize environment time
        self.env_time = 0

        # Handle the first event if it occurs at the start of the simulation
        failure = 0
        if handle_first_events and self.event_buffer and self.event_buffer[0].time == self.env_time:
            event = self.event_buffer.pop(0)
            failure, self.next_bike_id = event_handler(
                event=event,
                station_dict=self.stations,
                nearby_nodes_dict=self.nearby_nodes_dict,
                distance_matrix=self.distance_matrix,
                system_bikes=self.system_bikes,
                outside_system_bikes=self.outside_system_bikes,
                logger=self.logger,
                next_bike_id=self.next_bike_id
            )

        return failure


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


    def _jump_to_next_state(self, steps: int = 0) -> list:
        failures = []

        # Iterate through each step
        for _ in range(0, steps):
            # print("Entering in for")
            # Increment the environment time by 30 seconds
            self.env_time += 30

            total_failures = 0
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
                    logger=self.logger,
                    next_bike_id=self.next_bike_id
                )
                total_failures += failure
            failures.append(total_failures)

            # Log the updated state after processing events
            self.logger.log_state(
                step=int(self.env_time / 30),
                time=convert_seconds_to_hours_minutes((self.timeslot * 3 + 1) * 3600 + self.env_time)
            )

        # Return the total number of failures encountered
        return failures


    def _get_obs(self) -> np.array:
        # FIXME: Fix the observation space
        # Encode time slot and day
        h, _ = divmod((self.timeslot * 3 + 1) * 3600 + self.env_time, 3600)
        hour = [1 if h == i else 0 for i in range(24)]
        day = [1 if self.day == d else 0 for d in days2num.keys()]

        # Combine all features into a single observation array
        observation = np.array(
            [self.truck.get_load() / self.truck.max_load]
            + day
            + hour
        )

        return observation.astype(np.float32)


    def _get_reward(self, steps: int, failures: list, distance: int) -> float:
        # Cost per distance traveled
        hour = divmod((self.timeslot * 3 + 1) * 3600 + self.env_time, 3600)[0] % 24
        distance_cost = (distance/1000)*self.consumption_matrix.loc[hour, self.day]

        # Maximum 100 bikes per region
        bike_per_region_cost = self._compute_bike_per_region_cost()

        # Maximum 2500 bikes in the system
        # total_bikes_cost = logistic_penalty_function(M=1, k=0.03, b=self.maximum_number_of_bikes, x=len(self.system_bikes))

        # Reward for each step
        reward = - steps - failures[0] - distance_cost - bike_per_region_cost
        for i, failure in enumerate(failures[1:]):
            reward -= failure*(self.discount_factor**i)

        return reward


    def _compute_bike_per_region_cost(self) -> float:
        total_cost = 0.0
        for cell_id, cell in self.cells.items():
            n_bikes = cell.get_total_bikes()
            cost = logistic_penalty_function(M=1, k=1, b=100, x=n_bikes)
            total_cost += cost
        return total_cost


    def _update_graph(self):
        """
        Update the attributes of the subgraph with regional metrics.

        Parameters:
            - subgraph (nx.Graph): The subgraph to update with regional metrics.
        """
        # FIXME: Fix the observation space
        for cell_id, cell in self.cells.items():
            center_node = cell.get_center_node()

            # Initialize regional metrics
            demand_rate, average_battery_level = 0.0, 0.0
            low_battery_ratio, variance_battery_level = 0.0, 0.0

            # Aggregate metrics for nodes in the cell
            for node in cell.nodes:
                bikes = self.stations[node].get_bikes()
                battery_levels = [bike.get_battery() / bike.get_max_battery() for bike in bikes.values()]

                # Update regional metrics
                demand_rate += self.stations[node].get_request_rate()
                if battery_levels:
                    average_battery_level += np.mean(battery_levels)
                    variance_battery_level += np.var(battery_levels)
                    low_battery_ratio += np.mean(
                        [level < 0.2 for level in battery_levels]
                    )

            # Avoid division by zero by ensuring at least one node
            num_nodes = max(1, len(cell.nodes))
            demand_rate /= self.global_rate
            average_battery_level /= num_nodes
            low_battery_ratio /= num_nodes
            variance_battery_level /= num_nodes

            # Update attributes in the subgraph
            if center_node in self.cell_subgraph:
                self.cell_subgraph.nodes[center_node]['demand_rate'] = demand_rate
                self.cell_subgraph.nodes[center_node]['average_battery_level'] = average_battery_level
                self.cell_subgraph.nodes[center_node]['low_battery_ratio'] = low_battery_ratio
                self.cell_subgraph.nodes[center_node]['variance_battery_level'] = variance_battery_level
                self.cell_subgraph.nodes[center_node]['total_bikes'] = cell.get_total_bikes() / len(self.system_bikes)


    def _get_truck_position(self) -> tuple[float, float]:
        truck_coords = self.nodes_dict.get(self.truck.get_position())
        normalized_coords = ((truck_coords[0] - self.min_lat) / (self.max_lat - self.min_lat),
                             (truck_coords[1] - self.min_lon) / (self.max_lon - self.min_lon))
        return normalized_coords


    def _get_system_data(self) -> tuple[dict, dict, dict]:
        return self.stations, self.system_bikes, self.outside_system_bikes


    def _adjust_depot_system_discrepancy(self):
        """
        Adjust the discrepancy of the depot + system bikes to the maximum number of bikes.
        This is done to prevent the truck from failing to load bikes from the depot.
        """
        depot_load = len(self.depot)
        system_load = len(self.system_bikes)

        if depot_load + system_load < self.maximum_number_of_bikes:
            n_bikes = self.maximum_number_of_bikes - depot_load - system_load
            for _ in range(n_bikes):
                bike = self.outside_system_bikes.pop(next(iter(self.outside_system_bikes)))
                self.depot[bike.get_bike_id()] = bike


    def close(self):
        """Clean up resources."""
        self.background_thread.join()


    def save(self):
        """Save the environment state."""
        return {
            "pmf_matrix": self.pmf_matrix,
            "global_rate": self.global_rate,
            "global_rate_dict": self.global_rate_dict,
            "system_bikes": self.system_bikes,
            "outside_system_bikes": self.outside_system_bikes,
            "maximum_number_of_bikes": self.maximum_number_of_bikes,
            "current_cell_id": self.current_cell_id,
            "stations": self.stations,
            "truck": self.truck,
            "event_buffer": self.event_buffer,
            "next_event_buffer": self.next_event_buffer,
            "env_time": self.env_time,
            "energy_cost_per_time": self.energy_cost_per_time,
            "timeslot": self.timeslot,
            "day": self.day,
            "cell_subgraph": self.cell_subgraph,
            "timeslots_completed": self.timeslots_completed,
            "days_completed": self.days_completed,
            "next_bike_id": self.next_bike_id,
            "total_timeslots": self.total_timeslots,
            "background_thread": self.background_thread,
            "logging": self.logging,
        }


    def load(self, state):
        """Load the environment state."""
        self.pmf_matrix = state["pmf_matrix"]
        self.global_rate = state["global_rate"]
        self.global_rate_dict = state["global_rate_dict"]
        self.system_bikes = state["system_bikes"]
        self.outside_system_bikes = state["outside_system_bikes"]
        self.maximum_number_of_bikes = state["maximum_number_of_bikes"]
        self.current_cell_id = state["current_cell_id"]
        self.stations = state["stations"]
        self.truck = state["truck"]
        self.event_buffer = state["event_buffer"]
        self.next_event_buffer = state["next_event_buffer"]
        self.env_time = state["env_time"]
        self.energy_cost_per_time = state["energy_cost_per_time"]
        self.timeslot = state["timeslot"]
        self.day = state["day"]
        self.cell_subgraph = state["cell_subgraph"]
        self.timeslots_completed = state["timeslots_completed"]
        self.days_completed = state["days_completed"]
        self.next_bike_id = state["next_bike_id"]
        self.total_timeslots = state["total_timeslots"]
        self.background_thread = state["background_thread"]
        self.logging = state["logging"]
        self.logger.set_logging(self.logging)