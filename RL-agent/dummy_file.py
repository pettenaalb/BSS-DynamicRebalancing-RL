import torch
import argparse
import pickle
import os

import gymnasium as gym
import numpy as np

import gymnasium_env.register_env

from tqdm.contrib.telegram import tqdm as tqdm_telegram
from tqdm import tqdm
from utils import convert_graph_to_data, convert_seconds_to_hours_minutes, Actions

# ----------------------------------------------------------------------------------------------------------------------

data_path = "../data/"

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    # "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# set seed
seed = 31
np.random.seed(seed)
torch.manual_seed(seed)

params = {
    "num_episodes": 10,                 # Total number of training episodes
    "batch_size": 256,                  # Batch size for replay buffer sampling
    "replay_buffer_capacity": 1e5,      # Capacity of replay buffer: 1 million transitions
    "gamma": 0.99,                      # Discount factor
    "epsilon_start": 1.0,               # Starting exploration rate
    "epsilon_delta": 0.05,
    "epsilon_end": 0.00,                # Minimum exploration rate
    "epsilon_decay": 500,               # Epsilon decay rate
    "lr": 1e-3,                         # Learning rate
    "total_timeslots": 56,             # Total number of time slots in one episode (1 month)
    "maximum_number_of_bikes": 300,     # Maximum number of bikes in the system
}

enable_telegram = False
BOT_TOKEN = '7911945908:AAHkp-x7at3fIadrlmahcTB1G6_ni2awbt4'
CHAT_ID = '16830298'

def train_dueling_dqn(env: gym, episode: int, tbar: tqdm | tqdm_telegram) -> dict:
    """
    Trains a Dueling Deep Q-Network agent using experience replay.

    Parameters:
        - agent (DQNAgent): The Dueling DQN agent to train.
        - num_episodes (int): The number of episodes to train the agent.
        - batch_size (int): The batch size for training the agent.

    Returns:
        - rewards_per_timeslot (list): The rewards obtained per time slot during training.
        - failures_per_timeslot (list): The failures per time slot during training.
    """

    # Initialize episode metrics
    timeslot = 0
    timeslots_completed = 0
    total_failures = 0
    failures_per_timeslot = []

    # Reset environment and agent state
    options ={
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'discount_factor': params["gamma"],
        'logging': False,
        'depot_id': 7,         # 491 back
        'initial_cell': 7,     # 185 back
    }

    agent_state, info = env.reset(options=options)

    not_done = True
    past_action = 0

    while not_done:

        # loop from 0 to 7
        action = Actions.STAY.value
        past_action = (past_action + 1) % 8

        # Step the environment with the chosen action
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        total_failures += sum(info['failures'])

        # Check if the episode is complete
        not_done = not done

        if timeslot_terminated:
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
    torch.cuda.empty_cache()

    return {'failures': failures_per_timeslot}

# ----------------------------------------------------------------------------------------------------------------------

def main():
    # Create the environment
    env = gym.make('gymnasium_env/BostonCity-v0', data_path=data_path)
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
        results = train_dueling_dqn(env, episode, tbar)
        total_failures.extend(results['failures'])

    tbar.close()

    with open('../results/total_failures_baseline.pkl', 'wb') as f:
        pickle.dump(total_failures, f)

    # Print the rewards after training
    print("\nTraining completed.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dummy file')
    parser.add_argument('--data_path', type=str, default="../data/", help='Path to the data folder')

    args = parser.parse_args()
    if args.data_path:
        data_path = args.data_path

    main()
