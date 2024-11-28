import gymnasium as gym
import numpy as np
import pandas as pd
import osmnx as ox

from enum import Enum
from gymnasium import spaces
from gymnasium.utils import seeding

from gymnasium_env.simulator.bike import Bike
from gymnasium_env.simulator.truck import Truck
from gymnasium_env.simulator.utils import initialize_graph, initialize_stations, load_cells_from_csv, kahan_sum

params = {
    'graph_file': 'cambridge_network.graphml',
    'cell_file': 'cell-data.csv',
    'distance_matrix_file': 'distance-matrix.csv',
    'filtered_stations_file': 'filtered_stations.csv',
    'matrices_folder': 'matrices/09-10',
    'rates_folder': 'rates/09-10',
    'trips_folder': 'trips/'
}

# ----------------------------------------------------------------------------------------------------------------------

class Actions(Enum):
    STAY = 0
    RIGHT = 1
    UP = 2
    LEFT = 3
    DOWN = 4
    DROP_BIKE = 5
    PICK_UP_BIKE = 6
    CHARGE_BIKE = 7

# ----------------------------------------------------------------------------------------------------------------------

class BostonCity(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    # Initialize the environment
    def __init__(self, data_path: str):
        super().__init__()

        self.data_path = data_path

        # Initialize the graph
        self.graph = initialize_graph(data_path + params['graph_file'])

        # Initialize the nodes dictionary
        nodes_gdf = ox.graph_to_gdfs(self.graph, edges=False)
        self.nodes_dict = {node_id: (row['y'], row['x']) for node_id, row in nodes_gdf.iterrows()}

        # Initialize the cells
        self.cells = load_cells_from_csv(data_path + params['cell_file'])

        # Initialize distance matrix
        self.distance_matrix = pd.read_csv(data_path + params['distance_matrix_file'], index_col='osmid')
        self.distance_matrix.index = self.distance_matrix.index.astype(int)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(int)

        # Define action_space
        self.action_space = spaces.Discrete(len(Actions))

        # Define observation_space
        # 5 features per region: [Number of bikes, Demand rate, Average battery level, Low-battery ratio, Variance of battery level]
        # 3 agent features: [Agent's position, Agent's load]
        self.num_regions = len(self.cells)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_regions * 5 + 3,),  # 4 features per region + 3 agent features
            dtype=np.float32
        )

        # Initialize other variables
        self.pmf_matrix, self.global_rate = None, None
        self.stations = None
        self.system_bikes, self.outside_system_bikes = None, None
        self.current_cell_id = None
        self.truck = None


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
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Reset RNG with seed

        # Initialize the environment to a starting state
        # Load the PMF matrix and global rate for a given day and time slot
        self.pmf_matrix, self.global_rate = self.load_pmf_matrix(options['day'], options['time_slot'])

        # Initialize stations
        # BIKES PER STATION MAP TO DO
        self.stations, self.system_bikes = initialize_stations(self.graph, pmf_matrix=self.pmf_matrix, global_rate=self.global_rate)
        self.outside_system_bikes = {}

        # Set the request rate for each station
        for cell in self.cells.values():
            for node in cell.nodes:
                self.stations[node].set_cell(cell)
                cell.set_total_bikes(cell.get_total_bikes() + self.stations[node].get_number_of_bikes())

        # Initialize truck
        self.current_cell_id = options['initial_cell']
        cell = self.cells[self.current_cell_id]
        bikes = {}
        for i in range(30):
            bikes[i] = Bike()
        self.truck = Truck(cell.center_node, cell, max_load=options['max_truck_load'], bikes=options['truck_load'])

        # Return initial observation and optional info dictionary
        return self.__get_obs(), self.__get_info()


    # Step: Update environment based on action, return obs, reward, done, truncated, info.
    def step(self, action):
        # Apply action to the environment
        # Compute observation, reward, step, terminated, truncated, info
        # Return these values
        pass


    # Render: Visualize the environment's current state.
    def render(self, mode="ansi"):
        # Implement rendering logic for the chosen mode
        pass


    # Close: Clean up any resources used by the environment.
    def close(self):
        pass


    def __get_obs(self) -> np.array:
        # Observation: Compute the observation based on the current state of the environment
        total_bikes_per_region = []
        demand_rate_per_region = []
        average_battery_level_per_region = []
        low_battery_ratio_per_region = []
        variance_battery_level_per_region = []
        for cell_id, cell in self.cells.items():
            demand_rate = 0
            average_battery_level = 0
            low_battery_ratio = 0
            variance_battery_level = 0
            for node in cell.nodes:
                bikes = self.stations[node].get_bikes()
                demand_rate += self.stations[node].get_request_rate()
                average_battery_level += np.mean([bike.get_battery() for bike in bikes])
                variance_battery_level += np.var([bike.get_battery() for bike in bikes])
                low_battery_ratio += np.mean([bike.get_battery()/bike.get_max_battery() < 0.2 for bike in bikes])
            demand_rate /= self.global_rate
            average_battery_level /= len(cell.nodes)
            low_battery_ratio /= len(cell.nodes)
            variance_battery_level /= len(cell.nodes)

            total_bikes_per_region.append(cell.get_total_bikes() / len(self.system_bikes))
            demand_rate_per_region.append(demand_rate)
            average_battery_level_per_region.append(average_battery_level)
            low_battery_ratio_per_region.append(low_battery_ratio)
            variance_battery_level_per_region.append(variance_battery_level)

        observation = np.array(
            total_bikes_per_region
            + demand_rate_per_region
            + average_battery_level_per_region
            + low_battery_ratio_per_region
            + variance_battery_level_per_region
            + [self.truck.get_position(), self.truck.get_load()]
        )

        return observation

    def __get_info(self):
        pass