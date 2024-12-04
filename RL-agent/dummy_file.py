import gymnasium as gym
import gymnasium_env
import time

from enum import Enum
from tqdm import tqdm

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

    env.reset(options=options)

    is_not_done = True
    while is_not_done:
        action = np.random.randint(0, 8)
        observation, reward, terminated, _, info = env.step(action)
        is_not_done = not terminated
        # env.render()
        # # pause for 2 seconds
        # time.sleep(2)


if __name__ == '__main__':
    main()