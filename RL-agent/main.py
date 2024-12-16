import torch
import matplotlib
import gymnasium_env.register_env

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from agent import DQNAgent
from utils import convert_graph_to_data, convert_seconds_to_hours_minutes, plot_data_online, plot_graph_with_truck_path
from replay_memory import ReplayBuffer
from torch_geometric.data import Data

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

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
    "epsilon_end": 0.01,                # Minimum exploration rate
    "epsilon_decay": 500,               # Epsilon decay rate
    "lr": 1e-3                          # Learning rate
}

days2num = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}

# ----------------------------------------------------------------------------------------------------------------------

def train_dueling_dqn(agent: DQNAgent, num_episodes: int, batch_size: int) -> tuple[list, list]:
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

    # Initialize metrics
    rewards_per_time_slot = []
    failures_per_time_slot = []

    for episode in range(num_episodes):
        # Reset environment and agent state
        agent_state, info = env.reset()

        state = convert_graph_to_data(info['cells_subgraph'])
        state.agent_state = np.concatenate([info['agent_position'], agent_state])
        state.steps = info['steps']

        graph = info['network_graph']
        cell_dict = info['cell_dict']
        nodes_dict = info['nodes_dict']

        # Initialize episode metrics
        total_reward = 0
        total_failures = 0
        not_done = True
        action_bins = [0] * action_size
        action_bin_labels = ['STAY', 'RIGHT', 'UP', 'LEFT', 'DOWN', 'DROP_BIKE', 'PICK_UP_BIKE', 'CHARGE_BIKE']
        truck_path = []
        time_slot = 0

        # Progress bar for the episode
        total_timeslots = 4 * 7 * 8  # Total time slots in a week
        tbar = tqdm(
            range(total_timeslots),
            desc=f"Episode {episode + 1}/{num_episodes}",
            position=0,
            leave=True,
            dynamic_ncols=True
        )

        while not_done:
            # Prepare the state for the agent
            single_state = Data(
                x=state.x,
                edge_index=state.edge_index,
                edge_attr=state.edge_attr,
                agent_state=torch.tensor(state.agent_state, dtype=torch.float32, device=device).unsqueeze(dim=0),
                batch=torch.zeros(state.x.size(0), dtype=torch.long, device=device)
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
                # Update target network periodically
                agent.update_target_network()

                # Record metrics for the current time slot
                rewards_per_time_slot.append(total_reward)
                failures_per_time_slot.append(total_failures)

                # Reset time slot metrics
                total_reward = 0

                # Log progress
                time = info['time']
                day = info['day']
                week = info['week']
                print(f"\rProcessing... Week {week}, {day.capitalize()}, {convert_seconds_to_hours_minutes(time)}",
                      end="", flush=True)

                # Online plot updates
                append_path = '_' + str(week) + '_' + day + '_' + str(time_slot)
                plot_data_online(rewards_per_time_slot, idx=1, xlabel='Time Slot', ylabel='Reward',
                                 save_path='../results/rewards/rewards' + append_path + '.png')
                plot_data_online(failures_per_time_slot, idx=2, xlabel='Time Slot', ylabel='Failures',
                                 save_path='../results/failures/failures' + append_path + '.png')
                plot_data_online(action_bins, idx=3, xlabel='Action', ylabel='Frequency', show_histogram=True,
                                 bin_labels=action_bin_labels,
                                 save_path='../results/actions/actions' + append_path + '.png')
                plot_graph_with_truck_path(graph, cell_dict, nodes_dict, truck_path, show_result=False, idx=4,
                                           save_path='../results/truck_paths/truck_paths' + append_path + '.png')

                truck_path = []

                time_slot = 0 if time_slot == 7 else time_slot + 1

                # Update progress bar
                tbar.update(1)

    return rewards_per_time_slot, failures_per_time_slot

# ----------------------------------------------------------------------------------------------------------------------

def main():
    # Set up replay buffer
    replay_buffer = ReplayBuffer(params["replay_buffer_capacity"], device)

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
    rewards_per_time_slot, failures_per_time_slot = train_dueling_dqn(
        agent,
        num_episodes=params["num_episodes"],
        batch_size=params["batch_size"]
    )

    # Print the rewards after training
    print("Training completed.")
    print(f"Rewards per episode: {rewards_per_time_slot}")


if __name__ == '__main__':
    main()