import os
import pickle
import torch
import argparse
import gc

import gymnasium_env.register_env

import gymnasium as gym
import numpy as np

from tqdm.contrib.telegram import tqdm as tqdm_telegram
from tqdm import tqdm
from agent import DQNAgent
from utils import convert_graph_to_data, convert_seconds_to_hours_minutes, send_telegram_message, Actions
from replay_memory import ReplayBuffer
from torch_geometric.data import Data

# ----------------------------------------------------------------------------------------------------------------------

data_path = "../data/"

env = gym.make('gymnasium_env/BostonCity-v0', data_path=data_path)
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
    "num_episodes": 24,                 # Total number of training episodes
    "batch_size": 256,                  # Batch size for replay buffer sampling
    "replay_buffer_capacity": 1e6,      # Capacity of replay buffer: 1 million transitions
    "gamma": 0.99,                      # Discount factor
    "epsilon_start": 1.0,               # Starting exploration rate
    "epsilon_delta": 0.05,
    "epsilon_end": 0.00,                # Minimum exploration rate
    "epsilon_decay": 500,               # Epsilon decay rate
    "lr": 1e-3,                         # Learning rate
    "total_timeslots": 224,             # Total number of time slots in two years
    "maximum_number_of_bikes": 3500,    # Maximum number of bikes in the system
}

enable_telegram = False
BOT_TOKEN = '7911945908:AAHkp-x7at3fIadrlmahcTB1G6_ni2awbt4'
CHAT_ID = '16830298'

enable_checkpoint = False
restore_from_checkpoint = False
enable_logging = False

# ----------------------------------------------------------------------------------------------------------------------

def train_dueling_dqn(agent: DQNAgent, batch_size: int, episode: int, tbar: tqdm | tqdm_telegram) -> dict:
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
    rewards_per_timeslot = []
    total_reward = 0
    failures_per_timeslot = []
    total_failures = 0
    q_values_per_timeslot = []
    action_per_step = []
    truck_path = []
    truck_path_per_timeslot = []

    inner_tbar = tqdm(
        range(360),
        desc="Timeslot training",
        position=1,
        leave=False,
        dynamic_ncols=True
    )

    # Reset environment and agent state
    options ={
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'discount_factor': params["gamma"],
        'logging': enable_logging
    }

    agent_state, info = env.reset(options=options)

    state = convert_graph_to_data(info['cells_subgraph'])
    state.agent_state = np.concatenate([info['agent_position'], agent_state])
    state.steps = info['steps']

    not_done = True

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
        avoid_action = None

        if info['number_of_system_bikes'] > params["maximum_number_of_bikes"] - 50:
            # Avoid dropping bikes if the system is almost full
            avoid_action = Actions.DROP_BIKE.value

        if info['number_of_system_bikes'] < 50:
            # Avoid picking up bikes if the system is almost empty
            avoid_action = Actions.PICK_UP_BIKE.value

        action = agent.select_action(single_state, avoid_action=avoid_action)
        action_per_step.append((action, agent.epsilon))

        # Step the environment with the chosen action
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # Update state with new information
        next_state = convert_graph_to_data(info['cells_subgraph'])
        next_state.agent_state = np.concatenate([info['agent_position'], agent_state])
        next_state.steps = info['steps']

        # Store the transition in the replay buffer
        agent.replay_buffer.push(state, action, reward, next_state, done)

        # Train the agent with a batch from the replay buffer
        agent.train_step(batch_size)

        # Update the state and metrics
        state = next_state
        total_reward += reward
        total_failures += sum(info['failures'])
        truck_path.append(info['path'])

        # Check if the episode is complete
        not_done = not done

        if timeslot_terminated:
            timeslots_completed += 1

            # Update target network periodically
            agent.update_target_network()
            if timeslots_completed % 200 == 0: # every 30 days
                agent.update_epsilon(delta_epsilon=params["epsilon_delta"])

            # Record metrics for the current time slot
            rewards_per_timeslot.append((total_reward/360, agent.epsilon))
            failures_per_timeslot.append((total_failures, agent.epsilon))
            truck_path_per_timeslot.append(truck_path)

            # Get Q-values for the current state
            with torch.no_grad():
                q_values = agent.get_q_values(single_state)
                q_values_per_timeslot.append((q_values.squeeze().cpu().numpy(), agent.epsilon))
                del q_values

            # Log progress
            tbar.set_description(f"Episode {episode}, Week {info['week'] % 52}, {info['day'].capitalize()} "
                                 f"at {convert_seconds_to_hours_minutes(info['time'])}")

            # Reset time slot metrics
            total_reward = 0
            total_failures = 0
            truck_path = []
            timeslot = 0 if timeslot == 7 else timeslot + 1

            inner_tbar.reset()

            # Update progress bar
            tbar.set_postfix({'epsilon': agent.epsilon, 'failures': failures_per_timeslot[-1][0]})
            tbar.update(1)

        # Explicitly delete single_state
        del single_state

        inner_tbar.update(info['steps']+1)

    env.close()
    inner_tbar.close()
    torch.cuda.empty_cache()

    results = {
        "rewards_per_timeslot": rewards_per_timeslot,
        "failures_per_timeslot": failures_per_timeslot,
        "q_values_per_timeslot": q_values_per_timeslot,
        "action_per_step": action_per_step
    }

    return results

# ----------------------------------------------------------------------------------------------------------------------

def save_checkpoint(main_variables: dict, agent: DQNAgent, buffer: ReplayBuffer):
    checkpoint_path = data_path + 'checkpoints/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print(f"Directory '{checkpoint_path}' created.")

    with open(checkpoint_path + 'main_variables.pkl', 'wb') as f:
        pickle.dump(main_variables, f)
    agent.save_checkpoint(checkpoint_path + 'agent.pt')
    buffer.save_to_files(checkpoint_path + 'replay_buffer/')

    pass


def restore_checkpoint(agent: DQNAgent, buffer: ReplayBuffer) -> dict:
    checkpoint_path = data_path + 'checkpoints/'

    with open(checkpoint_path + 'main_variables.pkl', 'rb') as f:
        main_variables = pickle.load(f)
    agent.load_checkpoint(checkpoint_path + 'agent.pt')
    buffer.load_from_files(checkpoint_path + 'replay_buffer/')

    return main_variables

# ----------------------------------------------------------------------------------------------------------------------

def main():
    print(f"Device in use: {device}\n")
    # Set up replay buffer
    replay_buffer = ReplayBuffer(params["replay_buffer_capacity"])

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
        # Progress bar for the episode
        if enable_telegram:
            tbar = tqdm_telegram(
                range(params["total_timeslots"]*params["num_episodes"]),
                desc="Training Episode, Week 1, Monday at 01:00:00",
                position=0,
                leave=True,
                dynamic_ncols=True,
                token=BOT_TOKEN,
                chat_id=CHAT_ID
            )
        else:
            tbar = tqdm(
                range(params["total_timeslots"]*params["num_episodes"]),
                desc="Training Year 1, Week 1, Monday at 01:00:00",
                position=0,
                leave=True,
                dynamic_ncols=True
            )

        # Restore from checkpoint
        starting_episode = 0
        if restore_from_checkpoint:
            main_variables = restore_checkpoint(agent, replay_buffer)
            starting_episode = main_variables['episode'] + 1
            tbar = main_variables['tbar']

        for episode in range(starting_episode, params["num_episodes"]):
            results = train_dueling_dqn(agent, params["batch_size"], episode, tbar)

            # Save result lists
            results_path = '../results/training/data/'+ str(episode).zfill(2) + '/'
            if not os.path.exists(results_path):
                os.makedirs(results_path)
                print(f"Directory '{results_path}' created.")

            with open(results_path + 'rewards_per_timeslot.pkl', 'wb') as f:
                pickle.dump(results['rewards_per_timeslot'], f)
            with open(results_path + 'failures_per_timeslot.pkl', 'wb') as f:
                pickle.dump(results['failures_per_timeslot'], f)
            with open(results_path + 'q_values_per_timeslot.pkl', 'wb') as f:
                pickle.dump(results['q_values_per_timeslot'], f)
            with open(results_path + 'action_per_step.pkl', 'wb') as f:
                pickle.dump(results['action_per_step'], f)

            # Save checkpoint
            if enable_checkpoint:
                main_variables = {
                    'episode': episode,
                    'tbar': tbar,
                }
                save_checkpoint(main_variables, agent, replay_buffer)

            gc.collect()

        tbar.close()
    except Exception as e:
        if enable_telegram:
            send_telegram_message(f"An error occurred during training: {e}", BOT_TOKEN, CHAT_ID)
        raise e
    except KeyboardInterrupt:
        if enable_telegram:
            send_telegram_message("Training interrupted by user.", BOT_TOKEN, CHAT_ID)
        print("\nTraining interrupted by user.")
        return

    # Save the trained model
    trained_models_folder = data_path + 'trained_models'

    if not os.path.exists(trained_models_folder):
        os.makedirs(trained_models_folder)
        print(f"Directory '{trained_models_folder}' created.")

    agent.save_model(trained_models_folder + '/DuelingDQN.pt')

    # Print the rewards after training
    print("\nTraining completed.")
    send_telegram_message("Training completed.", BOT_TOKEN, CHAT_ID)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Dueling DQN agent.')
    parser.add_argument('--enable_telegram', action='store_true', help='Enable Telegram notifications.')
    parser.add_argument('--data_path', type=str, default='../data/', help='Path to the data folder.')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device to use.')
    parser.add_argument('--enable_logging', action='store_true', help='Enable logging.')
    parser.add_argument('--enable_checkpoint', action='store_true', help='Enable checkpointing.')
    parser.add_argument('--restore_from_checkpoint', action='store_true', help='Restore from checkpoint.')

    args = parser.parse_args()

    if args.enable_telegram:
        enable_telegram = True

    if args.data_path:
        data_path = args.data_path

    if args.cuda_device and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")

    if args.enable_logging:
        enable_logging = True

    if args.enable_checkpoint:
        enable_checkpoint = True

    if args.restore_from_checkpoint:
        restore_from_checkpoint = True

    main()
