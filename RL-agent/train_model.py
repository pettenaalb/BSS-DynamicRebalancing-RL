import os, shutil
import pickle
import threading
import torch
import argparse
import gc
import warnings

import gymnasium_env.register_env

import gymnasium as gym
import numpy as np

from tqdm.contrib.telegram import tqdm as tqdm_telegram
from tqdm import tqdm
from agent import DQNAgent
from prioritized_agent import PrioritizedDQNAgent
from utils import convert_graph_to_data, convert_seconds_to_hours_minutes, send_telegram_message, Actions
from replay_memory import ReplayBuffer
from prioritized_replay_memory import PrioritizedReplayBuffer
from torch_geometric.data import Data

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
    "num_episodes": 4,                              # Total number of training episodes
    "batch_size": 64,                               # Batch size for replay buffer sampling
    "replay_buffer_capacity": int(1e4),             # Capacity of replay buffer: 0.1 million transitions
    "gamma": 0.99,                                  # Discount factor
    "epsilon_start": 1.0,                           # Starting exploration rate
    "epsilon_delta": 0.05,                          # Epsilon decay rate
    "epsilon_end": 0.00,                            # Minimum exploration rate
    "epsilon_decay": 1e-5,                          # Epsilon decay constant
    "lr": 1e-4,                                     # Learning rate
    "total_timeslots": 56,                          # Total number of time slots in one episode (1 month)
    "maximum_number_of_bikes": 300,                 # Maximum number of bikes in the system
    "standard_reward": True,                        # Use standard reward function
    "results_path": "../results/training/",         # Path to save results
    "exploring_episodes": 10,                       # Number of episodes to explore
    "alpha": 0.6,                                   # Alpha parameter for prioritized replay buffer
    "beta": 0.4,                                    # Beta parameter for prioritized replay buffer
}

reward_params = {
    'W_ZERO_BIKES': 1.0,
    'W_CRITICAL_ZONES': 1.0,
    'W_DROP_PICKUP': 0.9,
    'W_MOVEMENT': 0.7,
    'W_CHARGE_BIKE': 0.9,
    'W_STAY': 0.7,
}

enable_telegram = False
BOT_TOKEN = '7911945908:AAHkp-x7at3fIadrlmahcTB1G6_ni2awbt4'
CHAT_ID = '16830298'

enable_checkpoint = False
restore_from_checkpoint = False
enable_logging = False

# ----------------------------------------------------------------------------------------------------------------------

def train_dueling_dqn(env: gym, agent: DQNAgent | PrioritizedDQNAgent, batch_size: int, episode: int, tbar: tqdm | tqdm_telegram) -> dict:
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
    losses = []
    reward_tracking = [[] for _ in range(len(Actions))]
    epsilon_per_timeslot = []
    deployed_bikes = []

    # Reset environment and agent state
    options ={
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'discount_factor': params["gamma"],
        'logging': enable_logging,
        'depot_id': 18,         # 491 back
        'initial_cell': 18,     # 185 back
        'standard_reward': params["standard_reward"],
        'reward_params': reward_params,
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

        # Remove actions that are not allowed
        avoid_actions = []

        # Avoid dropping bikes if the system is almost full
        if info['number_of_system_bikes'] >= (params["maximum_number_of_bikes"] - 1):
            avoid_actions.append(Actions.DROP_BIKE.value)

        # Avoid moving in directions where the truck cannot move
        truck_adjacent_cells = info['truck_neighbor_cells']

        if truck_adjacent_cells['down'] is None:
            avoid_actions.append(Actions.DOWN.value)

        if truck_adjacent_cells['up'] is None:
            avoid_actions.append(Actions.UP.value)

        if truck_adjacent_cells['left'] is None:
            avoid_actions.append(Actions.LEFT.value)

        if truck_adjacent_cells['right'] is None:
            avoid_actions.append(Actions.RIGHT.value)

        # Select an action using the agent
        action = agent.select_action(single_state, avoid_action=avoid_actions)

        # Step the environment with the chosen action
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # Update state with new information
        next_state = convert_graph_to_data(info['cells_subgraph'])
        next_state.agent_state = np.concatenate([info['agent_position'], agent_state])
        next_state.steps = info['steps']

        # Store the transition in the replay buffer
        agent.replay_buffer.push(state, action, reward, next_state, done)

        # Train the agent with a batch from the replay buffer
        loss = agent.train_step(batch_size)

        # Update the state
        state = next_state

        # Update the metrics
        action_per_step.append(action)
        reward_tracking[action].append(reward)
        losses.append(loss)

        total_reward += reward
        total_failures += sum(info['failures'])

        # Check if the episode is complete
        not_done = not done

        agent.update_epsilon(steps_in_action=info['steps'])
        # agent.update_beta()

        if timeslot_terminated:
            timeslots_completed += 1

            # Update target network periodically
            # agent.update_target_network()
            # if timeslots_completed % int((params['exploring_episodes'])*params['total_timeslots']/20) == 0:
            #     agent.update_epsilon(delta_epsilon=params["epsilon_delta"])

            # Get Q-values for the current state
            with torch.no_grad():
                q_values = agent.get_q_values(single_state)
                q_values_per_timeslot.append(q_values.squeeze().cpu().numpy())
                del q_values

            # Record metrics for the current time slot
            rewards_per_timeslot.append(total_reward/360)
            failures_per_timeslot.append(total_failures)
            epsilon_per_timeslot.append(agent.epsilon)
            deployed_bikes.append(info['number_of_system_bikes'])

            # Reset time slot metrics
            total_reward = 0
            total_failures = 0
            timeslot = 0 if timeslot == 7 else timeslot + 1

            # Update progress bar
            tbar.set_description(f"Episode {episode}, Week {info['week'] % 52}, {info['day'].capitalize()} "
                                 f"at {convert_seconds_to_hours_minutes(info['time'])}")
            tbar.set_postfix({'epsilon': agent.epsilon, 'failures': failures_per_timeslot[-1]})
            tbar.update(1)

        # Explicitly delete single_state
        del single_state

    env.close()
    torch.cuda.empty_cache()

    results = {
        "rewards_per_timeslot": rewards_per_timeslot,
        "failures_per_timeslot": failures_per_timeslot,
        "q_values_per_timeslot": q_values_per_timeslot,
        "action_per_step": action_per_step,
        "losses": losses,
        "reward_tracking": reward_tracking,
        "epsilon_per_timeslot": epsilon_per_timeslot,
        "deployed_bikes": deployed_bikes,
    }

    return results

# ----------------------------------------------------------------------------------------------------------------------

def save_checkpoint(main_variables: dict, agent: DQNAgent | PrioritizedDQNAgent, buffer: ReplayBuffer | PrioritizedReplayBuffer):
    checkpoint_path = data_path + 'checkpoints/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        # print(f"Directory '{checkpoint_path}' created.")

    with open(checkpoint_path + 'main_variables.pkl', 'wb') as f:
        pickle.dump(main_variables, f)
    agent.save_checkpoint(checkpoint_path + 'agent.pt')
    buffer.save_to_files(checkpoint_path + 'replay_buffer/')

    print("Checkpoint saved.")


def restore_checkpoint(agent: DQNAgent | PrioritizedDQNAgent, buffer: ReplayBuffer | PrioritizedReplayBuffer) -> dict:
    print("Restoring checkpoint...", end=' ')
    checkpoint_path = data_path + 'checkpoints/'

    with open(checkpoint_path + 'main_variables.pkl', 'rb') as f:
        main_variables = pickle.load(f)
    print("Main variables loaded.", end=' ')
    agent.load_checkpoint(checkpoint_path + 'agent.pt')
    print("Agent loaded.", end=' ')
    buffer.load_from_files(checkpoint_path + 'replay_buffer/')
    print("Replay buffer loaded.")

    return main_variables

# ----------------------------------------------------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    print(f"Device in use: {device}\n")

    params["epsilon_decay"] = 0.5 * params["num_episodes"] * params["total_timeslots"]*180
    print(f"{params}\n")
    print(f"{reward_params}\n")

    # Create the environment
    env = gym.make('gymnasium_env/FullyDynamicEnv-v0', data_path=data_path)
    env.unwrapped.seed(seed)

    # Set up replay buffer
    replay_buffer = ReplayBuffer(params["replay_buffer_capacity"])
    # replay_buffer = PrioritizedReplayBuffer(params["replay_buffer_capacity"], params["alpha"])

    # Create background thread for checkpointing
    checkpoint_background_thread = None

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
    # agent = PrioritizedDQNAgent(
    #     replay_buffer=replay_buffer,
    #     num_actions=env.action_space.n,
    #     gamma=params["gamma"],
    #     epsilon_start=params["epsilon_start"],
    #     epsilon_end=params["epsilon_end"],
    #     epsilon_decay=params["epsilon_decay"],
    #     lr=params["lr"],
    #     device=device,
    #     beta=params["beta"],
    # )


    # Restore from checkpoint
    starting_episode = 0
    if restore_from_checkpoint:
        main_variables = restore_checkpoint(agent, replay_buffer)
        starting_episode = main_variables['episode'] + 1
        print(f"Restored from checkpoint. Resuming training from episode {starting_episode}.")

    # Train the agent using the training loop
    try:
        # Progress bar for the episode
        if enable_telegram:
            tbar = tqdm_telegram(
                range(params["total_timeslots"]*params["num_episodes"]),
                desc="Training Episode 1, Week 1, Monday at 01:00:00",
                initial=starting_episode*params["total_timeslots"],
                position=0,
                leave=True,
                dynamic_ncols=True,
                token=BOT_TOKEN,
                chat_id=CHAT_ID
            )
        else:
            tbar = tqdm(
                range(params["total_timeslots"]*params["num_episodes"]),
                desc="Training Episode 1, Week 1, Monday at 01:00:00",
                initial=starting_episode*params["total_timeslots"],
                position=0,
                leave=True,
                dynamic_ncols=True
            )

        if os.path.exists(str(params['results_path'])):
            shutil.rmtree(str(params['results_path']))

        for episode in range(starting_episode, params["num_episodes"]):
            # Train the agent for one episode
            results = train_dueling_dqn(env, agent, params["batch_size"], episode, tbar)

            # Save result lists
            results_path = str(params['results_path']) + 'data/'+ str(episode).zfill(2) + '/'
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            for key, value in results.items():
                with open(results_path + key + '.pkl', 'wb') as f:
                    pickle.dump(value, f)

            # Save checkpoint
            if enable_checkpoint:
                main_variables = {
                    'episode': episode,
                    'tbar_progress': tbar.n,
                }
                checkpoint_background_thread = threading.Thread(target=save_checkpoint, args=(main_variables, agent, replay_buffer))
                checkpoint_background_thread.start()

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

    # Wait for the background threads to finish
    if checkpoint_background_thread is not None:
        checkpoint_background_thread.join()

    # Print the rewards after training
    print("\nTraining completed.")
    if enable_telegram:
        send_telegram_message("Training completed.", BOT_TOKEN, CHAT_ID)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Dueling DQN agent.')
    parser.add_argument('--enable_telegram', action='store_true', help='Enable Telegram notifications.')
    parser.add_argument('--data_path', type=str, default='../data/', help='Path to the data folder.')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device to use.')
    parser.add_argument('--enable_logging', action='store_true', help='Enable logging.')
    parser.add_argument('--enable_checkpoint', action='store_true', help='Enable checkpointing.')
    parser.add_argument('--restore_from_checkpoint', action='store_true', help='Restore from checkpoint.')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to train.')
    parser.add_argument('--results_path', type=str, default='../results/training/', help='Path to save results.')
    parser.add_argument('--exploring_episodes', type=int, default=10, help='Number of episodes to explore.')

    args = parser.parse_args()

    # Assign variables based on the parsed arguments
    enable_telegram = args.enable_telegram
    data_path = args.data_path
    enable_logging = args.enable_logging
    enable_checkpoint = args.enable_checkpoint
    restore_from_checkpoint = args.restore_from_checkpoint
    params["num_episodes"] = args.num_episodes
    params["results_path"] = args.results_path
    params["exploring_episodes"] = args.exploring_episodes

    # Ensure the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

    # Set up the device
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    main()
