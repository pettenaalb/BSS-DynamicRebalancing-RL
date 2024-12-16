import gymnasium as gym
import networkx as nx
from twisted.mail.scripts.mailmail import failure

import gymnasium_env
import time

from enum import Enum
from tqdm import tqdm
from gymnasium_env.simulator.utils import plot_graph, convert_seconds_to_hours_minutes

import numpy as np

env = gym.make('gymnasium_env/BostonCity-v0', data_path='../data/')

class Actions(Enum):
    STAY = 0
    RIGHT = 1
    UP = 2
    LEFT = 3
    DOWN = 4
    DROP_BIKE = 5
    PICK_UP_BIKE = 6
    CHARGE_BIKE = 7

def main():
    options = {
        'day': 'monday',
        'time_slot': 2,
        'initial_cell': 185,
        'max_truck_load': 30
    }

    env.reset()

    is_not_done = True
    time = 0
    failures = 0
    while is_not_done:
        action = np.random.randint(0, 8)
        observation, reward, done, _, info = env.step(action)
        is_not_done = not done
        # env.render()
        # Check if edges and nodes are correctly initialized
        graph = info['cells_subgraph']
        truck_position = info['agent_position']
        time = info['time']
        day = info['day']
        week = info['week']
        print(f"\rProcessing... Week {week}, {day.capitalize()}, {convert_seconds_to_hours_minutes(time)}, {failures}", end="", flush=True)
        failures += info['failures']

        # edge_attrs = ['distance']
        # for u, v, k, attr in graph.edges(data=True, keys=True):
        #     for edge_attr in edge_attrs:
        #         if u == truck_position or v == truck_position:
        #             print(f'Edge {u} -> {v} ({k}): {attr[edge_attr]}')
        #
        # node_attrs = [
        #     'demand_rate_per_region',
        #     'average_battery_level_per_region',
        #     'low_battery_ratio_per_region',
        #     'variance_battery_level_per_region',
        #     'total_bikes_per_region'
        # ]
        # for attr in node_attrs:
        #     print(f'{attr}: {graph.nodes[truck_position][attr]}')

        # input("Press Enter to continue...")

    print(f'Time: {convert_seconds_to_hours_minutes(time)}')


if __name__ == '__main__':
    main()