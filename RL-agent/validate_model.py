import os
import pickle
import torch

import gymnasium_env.register_env

import gymnasium as gym
import numpy as np

from tqdm.contrib.telegram import tqdm as tqdm_telegram
from tqdm import tqdm
from agent import DQNAgent
from utils import convert_graph_to_data, convert_seconds_to_hours_minutes, send_telegram_message
from torch_geometric.data import Data

# ----------------------------------------------------------------------------------------------------------------------

env = gym.make('gymnasium_env/BostonCity-v0', data_path='../data/')
action_size = env.action_space.n

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    # "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# set seed
seed = 31
env.unwrapped.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

params = {
    "num_episodes": 1,                  # Total number of training episodes
    "total_timeslots": 240              # Total number of time slots in one month (30 days)
}

days2num = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}

enable_telegram = True
BOT_TOKEN = '7911945908:AAHkp-x7at3fIadrlmahcTB1G6_ni2awbt4'
CHAT_ID = '16830298'

# ----------------------------------------------------------------------------------------------------------------------

def validate_dueling_dqn(agent: DQNAgent) -> tuple[list, list]:
    """
    Trains a Dueling Deep Q-Network agent using experience replay.

    Parameters:
        - agent (DQNAgent): The Dueling DQN agent to train.
        - num_episodes (int): The number of episodes to train the agent.
        - batch_size (int): The batch size for training the agent.

    Returns:
        - rewards_per_time_slot (list): The rewards obtained per time slot during training.
        - failures_per_time_slot (list): The failures per time slot during training.
    """

    # Reset environment and agent state
    options ={
        'total_timeslots': params["total_timeslots"],
    }
    agent_state, info = env.reset(options=options)

    state = convert_graph_to_data(info['cells_subgraph'])
    state.agent_state = np.concatenate([info['agent_position'], agent_state])
    state.steps = info['steps']

    # Initialize episode metrics
    time_slot = 0
    total_timeslots = 0
    rewards_per_time_slot = []
    total_reward = 0
    failures_per_time_slot = []
    total_failures = 0
    action_per_step = []
    truck_path = []

    not_done = True

    # Progress bar for the episode
    if enable_telegram:
        tbar = tqdm_telegram(
            range(params["total_timeslots"]),
            desc="Year 1, Week 1, Monday at 01:00:00",
            position=0,
            leave=True,
            dynamic_ncols=True,
            token=BOT_TOKEN,
            chat_id=CHAT_ID
        )
    else:
        tbar = tqdm(
            range(params["total_timeslots"]),
            desc="Year 1, Week 1, Monday at 01:00:00",
            position=0,
            leave=True,
            dynamic_ncols=True
        )

    while not_done:
        # Prepare the state for the agent
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(dim=0).to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )

        # Select an action using the agent
        action = agent.select_action(single_state, greedy=True)
        action_per_step.append(action)

        # Step the environment with the chosen action
        agent_state, reward, done, time_slot_terminated, info = env.step(action)

        # Update state with new information
        next_state = convert_graph_to_data(info['cells_subgraph'])
        next_state.agent_state = np.concatenate([info['agent_position'], agent_state])

        # Update the state and metrics
        state = next_state
        total_reward += reward
        total_failures += info['failures']
        truck_path.append(info['path'])

        # Check if the episode is complete
        not_done = not done

        if time_slot_terminated:
            total_timeslots += 1

            # Record metrics for the current time slot
            rewards_per_time_slot.append(total_reward/360)
            failures_per_time_slot.append(total_failures)

            # Log progress
            time_elapsed = info['time']
            day = info['day']
            week = info['week'] % 52
            year = info['year']
            tbar.set_description(f"Year {year + 1}, Week {week}, {day.capitalize()} at {convert_seconds_to_hours_minutes(time_elapsed)}")

            # Reset time slot metrics
            total_reward = 0
            truck_path = []
            time_slot = 0 if time_slot == 7 else time_slot + 1

            # Save result lists
            results_path = '../results/validation/data/'
            with open(results_path + 'rewards_per_time_slot.pkl', 'wb') as f:
                pickle.dump(rewards_per_time_slot, f)
            with open(results_path + 'failures_per_time_slot.pkl', 'wb') as f:
                pickle.dump(failures_per_time_slot, f)
            with open(results_path + 'action_per_step.pkl', 'wb') as f:
                pickle.dump(action_per_step, f)

            # Update progress bar
            tbar.update(1)

    tbar.close()
    env.close()

    return rewards_per_time_slot, failures_per_time_slot

# ----------------------------------------------------------------------------------------------------------------------

def main():
    print(f"Device in use: {device}\n")

    results_path = '../results/validation/data'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f"Directory '{results_path}' created.")

    # Initialize the DQN agent
    agent = DQNAgent(
        num_actions=env.action_space.n,
        device=device,
    )
    agent.load_model('../data/trained_models/DuelingDQN.pt')

    # Train the agent using the training loop
    try:
        validate_dueling_dqn(agent)
    except Exception as e:
        if enable_telegram:
            send_telegram_message(f"An error occurred during validation: {e}", BOT_TOKEN, CHAT_ID)
        raise e
    except KeyboardInterrupt:
        if enable_telegram:
            send_telegram_message("Validation interrupted by user.", BOT_TOKEN, CHAT_ID)
        print("\nValidation interrupted by user.")
        return

    # Print the rewards after training
    print("Validation completed.")


if __name__ == '__main__':
    main()
