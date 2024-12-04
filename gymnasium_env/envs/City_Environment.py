import math
import pickle
import gymnasium_env
import matplotlib

import gymnasium as gym
import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

from gymnasium import spaces
from gymnasium.utils import seeding

from gymnasium_env.simulator.bike_simulator import simulate_environment, event_handler
from gymnasium_env.simulator.truck_simulator import move_up, move_down, move_left, move_right, drop_bike, pick_up_bike, charge_bike, stay

from gymnasium_env.simulator.bike import Bike
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.utils import (initialize_graph, initialize_stations, load_cells_from_csv, kahan_sum,
                                           convert_seconds_to_hours_minutes, Logger, Actions)

matplotlib.use('Qt5Agg')

params = {
    'graph_file': 'utils/cambridge_network.graphml',
    'cell_file': 'utils/cell-data.csv',
    'distance_matrix_file': 'utils/distance-matrix.csv',
    'filtered_stations_file': 'utils/filtered_stations.csv',
    'matrices_folder': 'matrices/09-10',
    'rates_folder': 'rates/09-10',
    'trips_folder': 'trips/',
    'nearby_nodes': 'utils/nearby_nodes.pkl'
}

days = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}

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
        self.cells = load_cells_from_csv(data_path + params['cell_file'])

        # Initialize the distance matrix
        self.distance_matrix = pd.read_csv(data_path + params['distance_matrix_file'], index_col='osmid')
        self.distance_matrix.index = self.distance_matrix.index.astype(int)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(int)

        # Define action space
        self.action_space = spaces.Discrete(len(Actions))

        # Define observation space
        # Features: [5 per region + 3 agent features + 7 days + 24 hours]
        self.num_regions = len(self.cells)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_regions * 5 + 3 + 7 + 24,),
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

        # Set logging options
        self.logging = True
        self.logger.set_logging(self.logging)

        # Visualization elements
        self._fig, self._ax = None, None


    # Load the PMF matrix and global rate for a given day and time slot
    def load_pmf_matrix(self, day: str, time_slot: int) -> tuple[pd.DataFrame, float]:
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


    # Set the environment's seed
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # Reset: Return the initial observation and reset internal state.
    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        # Call parent class reset
        super().reset(seed=seed)

        # Update day and time slot if provided in options
        if options is not None:
            self.day = options['day']
            self.time_slot = options['time_slot']

        # Load PMF matrix and global rate for the current day and time slot
        self.pmf_matrix, self.global_rate = self.load_pmf_matrix(self.day, self.time_slot)

        # TODO: Implement initialization of the environment state with noise
        # Initialize stations and system bikes
        self.stations, self.system_bikes = initialize_stations(
            self.graph, pmf_matrix=self.pmf_matrix, global_rate=self.global_rate
        )
        self.outside_system_bikes = {}

        # Set the request rate for each station and update cell totals
        for cell in self.cells.values():
            for node in cell.nodes:
                self.stations[node].set_cell(cell)
                cell.set_total_bikes(cell.get_total_bikes() + self.stations[node].get_number_of_bikes())

        # Initialize the truck
        self.current_cell_id = options['initial_cell'] if options else 185
        cell = self.cells[self.current_cell_id]
        bikes = {i: Bike() for i in range(20)}
        max_truck_load = options['max_truck_load'] if options else 30
        self.truck = Truck(cell.center_node, cell, bikes=bikes, max_load=max_truck_load)

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
            distance_matrix=self.distance_matrix
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

        # Return the initial observation and an optional info dictionary
        return self._get_obs(), {}


    # Step: Update environment based on action, return obs, reward, terminated, truncated, info.
    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:
        # Log the start of the step
        self.logger.new_log_line()

        # Perform the action and log it
        t = 0
        if action == Actions.STAY.value:
            t = stay()
            self.logger.log_starting_action('STAY', t)
        elif action == Actions.RIGHT.value:
            t = move_right(self.truck, self.distance_matrix, self.cells)
            self.logger.log_starting_action('RIGHT', t)
        elif action == Actions.UP.value:
            t = move_up(self.truck, self.distance_matrix, self.cells)
            self.logger.log_starting_action('UP', t)
        elif action == Actions.LEFT.value:
            t = move_left(self.truck, self.distance_matrix, self.cells)
            self.logger.log_starting_action('LEFT', t)
        elif action == Actions.DOWN.value:
            t = move_down(self.truck, self.distance_matrix, self.cells)
            self.logger.log_starting_action('DOWN', t)
        elif action == Actions.DROP_BIKE.value:
            t = drop_bike(self.truck, self.distance_matrix)
            self.logger.log_starting_action('DROP_BIKE', t)
        elif action == Actions.PICK_UP_BIKE.value:
            t = pick_up_bike(self.truck, self.stations, self.distance_matrix)
            self.logger.log_starting_action('PICK_UP_BIKE', t)
        elif action == Actions.CHARGE_BIKE.value:
            t = charge_bike(self.truck, self.stations, self.distance_matrix)
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
            bike = self.truck.unload_bike()
            station.lock_bike(bike)

        # Log the ending action
        self.logger.log_ending_action(
            time=convert_seconds_to_hours_minutes((self.time_slot * 3 + 1) * 3600 + self.env_time)
        )

        # Perform a final state update
        failures += self._jump_to_next_state(steps=1)

        # Compute the outputs
        observation = self._get_obs()
        reward = self._get_reward(steps, failures)
        terminated = self.env_time >= 3600 * 3

        # Log truck state
        self.logger.log_truck(self.truck)

        # Prepare additional info
        info = {
            'steps': steps,
            'failures': failures
        }

        # Return the step results
        return observation, reward, terminated, False, info


    # Close: Clean up any resources used by the environment.
    def close(self):
        pass

    def _jump_to_next_state(self, steps: int = 0) -> int:
        total_failures = 0

        # Iterate through each step
        for _ in range(steps):
            # Increment the environment time by 30 seconds
            self.env_time += 30

            # Process all events that occurred before the updated environment time
            while self.event_buffer and self.event_buffer[0].time < self.env_time:
                event = self.event_buffer.pop(0)
                failures, self.system_bikes, self.outside_system_bikes = event_handler(
                    event,
                    self.stations,
                    self.nearby_nodes_dict,
                    self.distance_matrix,
                    self.system_bikes,
                    self.outside_system_bikes,
                    self.logger
                )
                total_failures += failures

            # Log the updated state after processing events
            self.logger.log_state(
                step=int(self.env_time / 30),
                time=convert_seconds_to_hours_minutes((self.time_slot * 3 + 1) * 3600 + self.env_time)
            )

        # Return the total number of failures encountered
        return total_failures


    def _get_obs(self) -> np.array:
        # FIXME: Fix the observation space
        # Initialize feature lists for regions
        total_bikes_per_region = []
        demand_rate_per_region = []
        average_battery_level_per_region = []
        low_battery_ratio_per_region = []
        variance_battery_level_per_region = []

        # Compute features for each region
        for cell_id, cell in self.cells.items():
            demand_rate, average_battery_level = 0, 0
            low_battery_ratio, variance_battery_level = 0, 0

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

            # Append normalized metrics for this region
            total_bikes_per_region.append(cell.get_total_bikes() / len(self.system_bikes))
            demand_rate_per_region.append(demand_rate)
            average_battery_level_per_region.append(average_battery_level)
            low_battery_ratio_per_region.append(low_battery_ratio)
            variance_battery_level_per_region.append(variance_battery_level)

        # Normalize truck coordinates
        truck_coords = self.nodes_dict.get(self.truck.get_position())
        lon_range, lat_range = self.max_lon - self.min_lon, self.max_lat - self.min_lat
        lon = (truck_coords[1] - self.min_lon) / lon_range if lon_range > 0 else 0
        lat = (truck_coords[0] - self.min_lat) / lat_range if lat_range > 0 else 0

        # Encode time slot and day
        h, _ = divmod((self.time_slot * 3 + 1) * 3600 + self.env_time, 3600)
        hour = [1 if h == i else 0 for i in range(24)]
        day = [1 if self.day == d else 0 for d in days.keys()]

        # Combine all features into a single observation array
        observation = np.array(
            total_bikes_per_region
            + demand_rate_per_region
            + average_battery_level_per_region
            + low_battery_ratio_per_region
            + variance_battery_level_per_region
            + [lon, lat, self.truck.get_load() / self.truck.max_load]
            + day
            + hour
        )

        return observation.astype(np.float32)

    def _get_reward(self, steps: int, failures: int) -> float:
        # FIXME: Fix the reward function
        # TODO: Implement a energy cost function
        return - steps - failures*steps - self.energy_cost_per_time*steps

    def render(self):
        # Initialize the figure and axes if not already done
        if not hasattr(self, '_fig') or self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(7, 7))

        # Clear the axis
        self._ax.clear()  # Clear the previous plot

        # Extract nodes and edges in WGS84 coordinates (lon, lat)
        nodes, edges = ox.graph_to_gdfs(self.graph, nodes=True, edges=True)

        # Convert cell_dict into a GeoDataFrame in WGS84 for easy plotting
        grid_geoms = [cell.boundary for cell in self.cells.values()]
        cell_gdf = gpd.GeoDataFrame(geometry=grid_geoms, crs="EPSG:4326")  # WGS84 CRS

        # Plot the graph edges in geographic coordinates
        edges.plot(ax=self._ax, linewidth=0.5, edgecolor="black", alpha=0.5)
        # Plot the graph nodes
        nodes.plot(ax=self._ax, markersize=2, color="black", alpha=0.5)

        # Overlay the grid cells
        cell_gdf.plot(ax=self._ax, linewidth=0.8, edgecolor="red", facecolor="blue", alpha=0.2)

        # Optionally plot center nodes and cell IDs
        for cell in self.cells.values():
            center_node = cell.center_node
            if center_node != 0:
                node_coords = self.graph.nodes[center_node]['x'], self.graph.nodes[center_node]['y']
                self._ax.plot(node_coords[0], node_coords[1], marker='o', color='yellow', markersize=4)

        # Truck position
        truck_coords = self.nodes_dict.get(self.truck.get_position())
        self._ax.plot(truck_coords[1], truck_coords[0], marker='o', color='red', markersize=10, label="Truck position")

        # Configure the plot appearance
        self._ax.axis('off')
        self._ax.legend()

        plt.xlim(truck_coords[1]-0.015, truck_coords[1]+0.015)
        plt.ylim(truck_coords[0]-0.015, truck_coords[0]+0.015)

        # Display the plot in a new window
        plt.draw()

        # Optional: Pause for a short time to ensure updates are visible (not necessary with plt.show())
        plt.pause(0.1)