import torch
import argparse
import pickle
import os

import gymnasium as gym
import numpy as np

import gymnasium_env.register_env

from tqdm.contrib.telegram import tqdm as tqdm_telegram
from tqdm import tqdm
from utils import convert_seconds_to_hours_minutes

# ----------------------------------------------------------------------------------------------------------------------

data_path = "../data/"

# set seed
seed = 31
np.random.seed(seed)
torch.manual_seed(seed)

params = {
    "num_episodes": 10,                 # Total number of training episodes
    "total_timeslots": 56,              # Total number of time slots in one episode (1 month)
    "maximum_number_of_bikes": 300,     # Maximum number of bikes in the system
}

def simulate_env(env: gym, episode: int, tbar: tqdm | tqdm_telegram) -> dict:
    # Initialize episode metrics
    timeslot = 0
    timeslots_completed = 0
    total_failures = 0
    failures_per_timeslot = []

    # Reset environment and agent state
    options ={
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'depot_id': 18,         # 491 back
        'initial_cell': 18,     # 185 back
    }

    env.reset(options=options)

    not_done = True

    while not_done:
        # Step the environment with the chosen action
        *_, done, terminated, info = env.step(0)

        total_failures += sum(info['failures'])

        # Check if the episode is complete
        not_done = not done

        if terminated:
            timeslots_completed += 1

            # Log progress
            tbar.set_description(f"Episode {episode}, Week {info['week'] % 52}, {info['day'].capitalize()} "
                                 f"at {convert_seconds_to_hours_minutes(info['time'])}")
            timeslot = 0 if timeslot == 7 else timeslot + 1

            failures_per_timeslot.append((total_failures, timeslot))

            # Update progress bar
            tbar.set_postfix({'failures': failures_per_timeslot[-1][0]})
            tbar.update(1)

            total_failures = 0

    env.close()

    return {'failures': failures_per_timeslot}

# ----------------------------------------------------------------------------------------------------------------------

def main():
    # Create the environment
    env = gym.make('gymnasium_env/StaticEnv-v0', data_path=data_path)
    env.unwrapped.seed(seed)

    tbar = tqdm(
        range(params["total_timeslots"]*params["num_episodes"]),
        desc="Training Episode 1, Week 1, Monday at 01:00:00",
        position=0,
        leave=True,
        dynamic_ncols=True
    )

    total_failures = []

    for episode in range(0, params["num_episodes"]):
        results = simulate_env(env, episode, tbar)
        total_failures.extend(results['failures'])

    tbar.close()

    with open('results/total_failures_baseline.pkl', 'wb') as f:
        pickle.dump(total_failures, f)

    # Print the rewards after training
    print("\nSimulation completed.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark file')
    parser.add_argument('--data_path', type=str, default="../data_cambridge_medium/", help='Path to the data folder')

    args = parser.parse_args()
    if args.data_path:
        data_path = args.data_path

    main()
