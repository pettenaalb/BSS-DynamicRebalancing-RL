import math
import pickle

import networkx as nx

import gymnasium_env
import matplotlib

import gymnasium as gym
import numpy as np
import pandas as pd
import osmnx as ox

from gymnasium import spaces
from gymnasium.utils import seeding

from gymnasium_env.simulator.bike_simulator import simulate_environment, event_handler
from gymnasium_env.simulator.truck_simulator import move_up, move_down, move_left, move_right, drop_bike, pick_up_bike, charge_bike, stay

from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.utils import (initialize_graph, initialize_stations, kahan_sum, Logger, Actions,
                                           convert_seconds_to_hours_minutes, initialize_cells_subgraph,
                                            plot_graph_with_grid, truncated_gaussian, initialize_bikes)

# matplotlib.use('Qt5Agg')

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
    metadata = {"render_modes": ["ansi"], "render_fps": 10}

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
        self.pmf_matrix, self.global_rate = None, None
        self.system_bikes, self.outside_system_bikes = None, None
        self.current_cell_id = None
        self.stations = None
        self.truck = None
        self.event_buffer = None
        self.env_time = 0
        self.energy_cost_per_time = 0
        self.time_slot = 0
        self.day = 'monday'
        self.cell_subgraph = None
        self.time_slots_completed = 0
        self.days_completed = 0
        self.next_bike_id = 0
        self.total_time_slots = 0

        # Set logging options
        self.logging = True
        self.logger.set_logging(self.logging)

        # Visualization elements
        self._fig, self._ax = None, None


    # Set the environment's seed
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # Reset: Return the initial observation and reset internal state.
    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        # Call parent class reset
        super().reset(seed=seed)

        # Update day and time slot if provided in options
        self.day = options.get('day', 'monday') if options else 'monday'
        self.time_slot = options.get('time_slot', 0) if options else 0

        self.total_time_slots = options.get('total_time_slots', 2*365*8) if options else 2*365*8

        # TODO: Implement initialization of the environment state with noise
        # Initialize stations and system bikes
        self.stations, self.system_bikes, self.outside_system_bikes, self.next_bike_id = initialize_stations(self.graph, next_bike_id=self.next_bike_id)

        # Set the number of bikes in each cell
        for cell in self.cells.values():
            for node in cell.nodes:
                self.stations[node].set_cell(cell)
                cell.set_total_bikes(cell.get_total_bikes() + self.stations[node].get_number_of_bikes())

        # Initialize the truck
        self.current_cell_id = options.get('initial_cell', 185) if options else 185
        cell = self.cells[self.current_cell_id]
        bikes, self.next_bike_id = initialize_bikes(n=15, next_bike_id=self.next_bike_id)
        max_truck_load = options.get('max_truck_load', 30) if options else 30
        self.truck = Truck(cell.center_node, cell, bikes=bikes, max_load=max_truck_load)

        # Initialize the day and time slot
        self._initialize_day_timeslot()

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
            'steps': 0
        }

        return observation, info


    # Step: Update environment based on action, return obs, reward, terminated, truncated, info.
    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:
        # Log the start of the step
        self.logger.new_log_line()

        # Perform the action and log it
        t = 0
        distance = 0
        prev_position = self.truck.get_position()
        hours, _ = divmod((self.time_slot * 3 + 1) * 3600 + self.env_time, 3600)
        hours = hours % 24
        mean_velocity = self.velocity_matrix.loc[hours, self.day]
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
            t, distance = drop_bike(self.truck, self.distance_matrix, mean_velocity)
            self.logger.log_starting_action('DROP_BIKE', t)
        elif action == Actions.PICK_UP_BIKE.value:
            t, distance, self.system_bikes = pick_up_bike(self.truck, self.stations, self.distance_matrix, mean_velocity,
                                                          self.depot_node, self.system_bikes)
            self.logger.log_starting_action('PICK_UP_BIKE', t)
        elif action == Actions.CHARGE_BIKE.value:
            t, distance, self.system_bikes = charge_bike(self.truck, self.stations, self.distance_matrix, mean_velocity,
                                                         self.depot_node, self.system_bikes)
            self.logger.log_starting_action('CHARGE_BIKE', t)

        # Calculate steps and log the state
        steps = math.ceil(t / 30)
        self.logger.log_state(
            step=int(self.env_time / 30),
            time=convert_seconds_to_hours_minutes((self.time_slot * 3 + 1) * 3600 + self.env_time)
        )

        # Update the environment state
        failures = self._jump_to_next_state(steps)

        # Handle specific actions post-environment update
        if action in {Actions.DROP_BIKE.value, Actions.CHARGE_BIKE.value}:
            station = self.stations.get(self.truck.get_position())
            try:
                bike = self.truck.unload_bike()
            except ValueError:
                distance = self.distance_matrix.loc[self.truck.get_position(), self.depot_node]
                velocity_kmh = truncated_gaussian(10, 70, mean_velocity, 5)
                t_reload = 2*int(distance * 3.6 / velocity_kmh)
                # CHECKME: is it ok to have, after the ride to the depot, only 15 bikes when the truck is empty?
                bikes, self.next_bike_id = initialize_bikes(n=15, next_bike_id=self.next_bike_id)
                self.truck.set_load(bikes)
                new_steps = math.ceil(t_reload+t / 30) - steps
                failures += self._jump_to_next_state(new_steps)
                bike = self.truck.unload_bike()
            station.lock_bike(bike)
            self.system_bikes[bike.get_bike_id()] = bike

        # Log the ending action
        self.logger.log_ending_action(
            time=convert_seconds_to_hours_minutes((self.time_slot * 3 + 1) * 3600 + self.env_time)
        )

        # Perform a final state update
        failures += self._jump_to_next_state(steps=1)

        # Log truck state
        self.logger.log_truck(self.truck)

        # Update the environment time and time slot
        terminated = False
        if self.env_time > 3600*3:
            residual_event_buffer = self.event_buffer
            for event in residual_event_buffer:
                event.time -= 3600*3
            if self.time_slot == 7:
                self.time_slot = 0
                self.day = num2days[(days2num[self.day] + 1) % 7]
                self.days_completed += 1
            else:
                self.time_slot += 1
            self._initialize_day_timeslot(residual_event_buffer)
            self.time_slots_completed += 1
            terminated = True


        # Compute the outputs
        observation = self._get_obs()
        reward = self._get_reward(steps, failures, distance)
        self._update_graph()
        path = (prev_position, self.truck.get_position())
        info = {
            'cells_subgraph': self.cell_subgraph,
            'agent_position': self._get_truck_position(),
            'steps': steps,
            'time': self.env_time + (self.time_slot * 3 + 1) * 3600,
            'day': self.day,
            'week': int(self.days_completed // 7),
            'year': int(self.days_completed // 365),
            'failures': failures,
            'path': path,
        }

        if self.time_slots_completed == self.total_time_slots:   # 2 years
            done = True
        else:
            done = False

        # Return the step results
        return observation, reward, done, terminated, info


    def render(self):
        truck_coords = self.nodes_dict.get(self.truck.get_position())
        x_lim = [truck_coords[1]-0.015, truck_coords[1]+0.015]
        y_lim = [truck_coords[0]-0.015, truck_coords[0]+0.015]
        total_graph = nx.compose(self.graph, self.cell_subgraph)
        plot_graph_with_grid(total_graph, self.cells, plot_center_nodes=True, plot_number_cells=False,
                             truck_coords=truck_coords, xlim=x_lim, ylim=y_lim)


    def _initialize_day_timeslot(self, residual_event_buffer: list = None):
        # Load PMF matrix and global rate for the current day and time slot
        self.pmf_matrix, self.global_rate = self._load_pmf_matrix(self.day, self.time_slot)

        for stn_id, stn in self.stations.items():
            stn.set_request_rate(kahan_sum(self.pmf_matrix.loc[stn_id])*self.global_rate)

        # Flatten the PMF matrix for event simulation
        values = self.pmf_matrix.values.flatten()
        ids = [(row, col) for row in self.pmf_matrix.index for col in self.pmf_matrix.columns]
        flattened_pmf = pd.DataFrame({'id': ids, 'value': values})
        flattened_pmf['cumsum'] = np.cumsum(flattened_pmf['value'].values)

        # Simulate the environment for the time slot
        self.event_buffer = simulate_environment(
            duration=3600 * 3,  # 3 hours
            time_slot=self.time_slot,
            global_rate=self.global_rate,
            pmf=flattened_pmf,
            stations=self.stations,
            distance_matrix=self.distance_matrix,
            residual_event_buffer=residual_event_buffer,
        )

        # Initialize environment time
        self.env_time = 0

        # Handle the first event if it occurs at the start of the simulation
        if self.event_buffer and self.event_buffer[0].time == self.env_time:
            event = self.event_buffer.pop(0)
            event_handler(
                event,
                self.stations,
                self.nearby_nodes_dict,
                self.distance_matrix,
                self.system_bikes,
                self.outside_system_bikes,
                logger=self.logger
            )


    # Load the PMF matrix and global rate for a given day and time slot
    def _load_pmf_matrix(self, day: str, time_slot: int) -> tuple[pd.DataFrame, float]:
        # Initialize the rate matrix
        matrix_path = self.data_path + params['matrices_folder'] + '/' + str(time_slot).zfill(2) + '/'
        pmf_matrix = pd.read_csv(matrix_path + day.lower() + '-pmf-matrix.csv', index_col='osmid')
        rate_matrix = pd.read_csv(matrix_path + day.lower() + '-rate-matrix.csv', index_col='osmid')

        # Convert index and columns to integers
        pmf_matrix.index = pmf_matrix.index.astype(int)
        pmf_matrix.columns = pmf_matrix.columns.astype(int)
        pmf_matrix.loc[10000, 10000] = 0.0

        global_rate = kahan_sum(rate_matrix.to_numpy().flatten())

        return pmf_matrix, global_rate


    def _jump_to_next_state(self, steps: int = 0) -> int:
        total_failures = 0

        # Iterate through each step
        for _ in range(steps):
            # Increment the environment time by 30 seconds
            self.env_time += 30

            # Process all events that occurred before the updated environment time
            while self.event_buffer and self.event_buffer[0].time < self.env_time:
                event = self.event_buffer.pop(0)
                failure, self.system_bikes, self.outside_system_bikes = event_handler(
                    event,
                    self.stations,
                    self.nearby_nodes_dict,
                    self.distance_matrix,
                    self.system_bikes,
                    self.outside_system_bikes,
                    self.logger
                )
                total_failures += failure

            # Log the updated state after processing events
            self.logger.log_state(
                step=int(self.env_time / 30),
                time=convert_seconds_to_hours_minutes((self.time_slot * 3 + 1) * 3600 + self.env_time)
            )

        # Return the total number of failures encountered
        return total_failures


    def _get_obs(self) -> np.array:
        # FIXME: Fix the observation space
        # Encode time slot and day
        h, _ = divmod((self.time_slot * 3 + 1) * 3600 + self.env_time, 3600)
        hour = [1 if h == i else 0 for i in range(24)]
        day = [1 if self.day == d else 0 for d in days2num.keys()]

        # Combine all features into a single observation array
        observation = np.array(
            [self.truck.get_load() / self.truck.max_load]
            + day
            + hour
        )

        return observation.astype(np.float32)


    def _get_reward(self, steps: int, failures: int, distance: int) -> float:
        # FIXME: Fix the reward function
        # TODO: bike limit per region * cost
        # TODO: normalize reward + positive reward (not necessary)
        hours, _ = divmod((self.time_slot * 3 + 1) * 3600 + self.env_time, 3600)
        hours = hours % 24
        mean_consumption = self.consumption_matrix.loc[hours, self.day]
        return - steps - failures*steps - (distance/1000)*mean_consumption


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
                self.cell_subgraph.nodes[center_node]['demand_rate_per_region'] = demand_rate
                self.cell_subgraph.nodes[center_node]['average_battery_level_per_region'] = average_battery_level
                self.cell_subgraph.nodes[center_node]['low_battery_ratio_per_region'] = low_battery_ratio
                self.cell_subgraph.nodes[center_node]['variance_battery_level_per_region'] = variance_battery_level
                self.cell_subgraph.nodes[center_node]['total_bikes_per_region'] = cell.get_total_bikes() / len(self.system_bikes)


    def _get_truck_position(self) -> tuple[float, float]:
        truck_coords = self.nodes_dict.get(self.truck.get_position())
        normalized_coords = ((truck_coords[0] - self.min_lat) / (self.max_lat - self.min_lat),
                             (truck_coords[1] - self.min_lon) / (self.max_lon - self.min_lon))
        return normalized_coords


    def _get_system_data(self) -> tuple[dict, dict, dict]:
        return self.stations, self.system_bikes, self.outside_system_bikes