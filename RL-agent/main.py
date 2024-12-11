import torch
import matplotlib

import gymnasium as gym
import gymnasium_env.register_env
import numpy as np

from tqdm import tqdm
from agent import DQNAgent
from utils import convert_graph_to_data, convert_seconds_to_hours_minutes
from replay_memory import ReplayBuffer
from torch_geometric.data import Data
from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------

env = gym.make('gymnasium_env/BostonCity-v0', data_path='../data/')
action_size = env.action_space.n

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

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
    "num_episodes": 1,  # Total number of training episodes
    "batch_size": 32,     # Batch size for replay buffer sampling
    "replay_buffer_capacity": 10000,  # Capacity of replay buffer
    "gamma": 0.99,  # Discount factor
    "epsilon_start": 1.0,  # Starting exploration rate
    "epsilon_end": 0.01,  # Minimum exploration rate
    "epsilon_decay": 500,  # Epsilon decay rate
    "lr": 1e-3  # Learning rate
}

days2num = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}

# ----------------------------------------------------------------------------------------------------------------------

def train_dueling_dqn(agent: DQNAgent, num_episodes: int, batch_size: int):
    rewards_per_time_slot = []

    for episode in range(num_episodes):
        agent_state, info = env.reset()
        network_state = convert_graph_to_data(info['cells_subgraph'])
        state = network_state
        state.agent_state = np.concatenate([info['agent_position'], agent_state])
        state.steps = info['steps']
        total_reward = 0

        not_done = True
        total_hours = 4*7*8
        tbar = tqdm(range(total_hours), desc=f"Episode {episode + 1}/{num_episodes}", position=0, leave=True, dynamic_ncols=True)
        while not_done:
            # Select an action and step in the environment
            batched_state = Data(
                x=state.x,
                edge_index=state.edge_index,
                edge_attr=state.edge_attr,
                agent_state=torch.tensor(state.agent_state, dtype=torch.float32, device=device),
                batch=torch.zeros(state.x.size(0), dtype=torch.long, device=device)
            )
            action = agent.select_action(batched_state)
            agent_state, reward, done, time_slot_terminated, info = env.step(action)
            network_state = convert_graph_to_data(info['cells_subgraph'])
            next_state = network_state
            next_state.agent_state = np.concatenate([info['agent_position'], agent_state])
            next_state.steps = info['steps']

            # Store experience in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train the agent
            agent.train_step(batch_size)

            # Update state and accumulate reward
            state = next_state
            total_reward += reward

            # Check if episode terminated
            not_done = not done

            if time_slot_terminated:
                agent.update_target_network()
                rewards_per_time_slot.append(total_reward)
                total_reward = 0
                time = info['time']
                day = info['day']
                week = info['week']
                print(f"\rProcessing... Week {week}, {day.capitalize()}, {convert_seconds_to_hours_minutes(time)}", end="", flush=True)
                tbar.update(1)

    return rewards_per_time_slot

# ----------------------------------------------------------------------------------------------------------------------

def plot_rewards(rewards, show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

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
    rewards_per_episode = train_dueling_dqn(
        agent,
        num_episodes=params["num_episodes"],
        batch_size=params["batch_size"]
    )

    # Print the rewards after training
    print("Training completed.")
    print(f"Rewards per episode: {rewards_per_episode}")


if __name__ == '__main__':
    main()