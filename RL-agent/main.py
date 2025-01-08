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
from replay_memory import ReplayBuffer
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
    "batch_size": 32,                   # Batch size for replay buffer sampling
    "replay_buffer_capacity": 10000,    # Capacity of replay buffer
    "gamma": 0.99,                      # Discount factor
    "epsilon_start": 1.0,               # Starting exploration rate
    "epsilon_delta": 0.05,
    "epsilon_end": 0.00,                # Minimum exploration rate
    "epsilon_decay": 500,               # Epsilon decay rate
    "lr": 1e-3,                         # Learning rate
    "total_timeslots": 5840             # Total number of time slots in two years
}

days2num = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}

enable_telegram = True
BOT_TOKEN = '7911945908:AAHkp-x7at3fIadrlmahcTB1G6_ni2awbt4'
CHAT_ID = '16830298'

# ----------------------------------------------------------------------------------------------------------------------

def train_dueling_dqn(agent: DQNAgent, batch_size: int) -> tuple[list, list]:
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
    q_values_per_time_slot = []
    action_bins = [0] * action_size
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
        action = agent.select_action(single_state)
        action_bins[action] += 1

        # Step the environment with the chosen action
        agent_state, reward, done, time_slot_terminated, info = env.step(action)
        network_state = convert_graph_to_data(info['cells_subgraph'])

        # Update state with new information
        next_state = network_state
        next_state.agent_state = np.concatenate([info['agent_position'], agent_state])
        next_state.steps = info['steps']

        # Store the transition in the replay buffer
        agent.replay_buffer.push(state, action, reward, next_state, done)

        # Train the agent with a batch from the replay buffer
        agent.train_step(batch_size)

        # Update the state and metrics
        state = next_state
        total_reward += reward
        total_failures += info['failures']
        truck_path.append(info['path'])

        # Check if the episode is complete
        not_done = not done

        if time_slot_terminated:
            total_timeslots += 1

            # Update target network periodically
            agent.update_target_network()
            if total_timeslots % 292 == 0:
                agent.update_epsilon(delta_epsilon=params["epsilon_delta"])

            # Record metrics for the current time slot
            rewards_per_time_slot.append(total_reward/360)
            failures_per_time_slot.append(total_failures)

            # Get Q-values for the current state
            with torch.no_grad():
                q_values = agent.get_q_values(single_state)
                q_values_per_time_slot.append(q_values.squeeze().cpu().numpy())

            # Log progress
            time_elapsed = info['time']
            day = info['day']
            week = info['week']
            year = info['year']
            # print(f"\rProcessing... Year {year + 1}, Week {week}, {day.capitalize()}, {convert_seconds_to_hours_minutes(time_elapsed)}",
            #       end="", flush=True)
            tbar.set_description(f"Year {year + 1}, Week {week}, {day.capitalize()} at {convert_seconds_to_hours_minutes(time_elapsed)}")

            # Reset time slot metrics
            total_reward = 0
            truck_path = []
            time_slot = 0 if time_slot == 7 else time_slot + 1

            # Save result lists
            results_path = '../results/data'
            with open(results_path + 'rewards_per_time_slot.pkl', 'wb') as f:
                pickle.dump(rewards_per_time_slot, f)
            with open(results_path + 'failures_per_time_slot.pkl', 'wb') as f:
                pickle.dump(failures_per_time_slot, f)
            with open(results_path + 'q_values_per_time_slot.pkl', 'wb') as f:
                pickle.dump(q_values_per_time_slot, f)
            with open(results_path + 'action_bins.pkl', 'wb') as f:
                pickle.dump(action_bins, f)

            # Update progress bar
            tbar.set_postfix({'epsilon': agent.epsilon})
            tbar.update(1)

    tbar.close()
    env.close()

    return rewards_per_time_slot, failures_per_time_slot

# ----------------------------------------------------------------------------------------------------------------------

def main():
    print(f"Device in use: {device}\n")
    # Set up replay buffer
    replay_buffer = ReplayBuffer(params["replay_buffer_capacity"], device)

    results_path = '../results/data'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f"Directory '{results_path}' created.")

    # Initialize the DQN agent
    agent = DQNAgent(
        replay_buffer=replay_buffer,
        num_actions=env.action_space.n,
        gamma=params["gamma"],
        epsilon_start=params["epsilon_start"],
        epsilon_end=params["epsilon_end"],
        epsilon_decay=params["epsilon_decay"],
        lr=params["lr"],
        device=device,
    )

    # Train the agent using the training loop
    try:
        rewards_per_time_slot, failures_per_time_slot = train_dueling_dqn(
            agent,
            batch_size=params["batch_size"]
        )
    except Exception as e:
        if enable_telegram:
            send_telegram_message(f"An error occurred during training: {e}", BOT_TOKEN, CHAT_ID)
        raise e
    except KeyboardInterrupt:
        if enable_telegram:
            send_telegram_message("Training interrupted by user.", BOT_TOKEN, CHAT_ID)
        print("\nTraining interrupted by user.")
        return

    # Print the rewards after training
    print("Training completed.")
    print(f"Rewards per episode: {rewards_per_time_slot}")


if __name__ == '__main__':
    main()
