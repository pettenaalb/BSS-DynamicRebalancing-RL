import pickle
import numpy as np

from utils import plot_data_online

mode = 'train'
action_bin_labels = ['STAY', 'RIGHT', 'UP', 'LEFT', 'DOWN', 'DROP_BIKE', 'PICK_UP_BIKE', 'CHARGE_BIKE']
epsilon_threshold = 0.1

def main():
    if mode == 'train':
        base_path = 'training/'
    elif mode == 'validate':
        base_path = 'validation/'
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'validate'.")

    with open(base_path + 'data/rewards_per_time_slot.pkl', 'rb') as f:
        rewards_per_time_slot = pickle.load(f)
    with open(base_path + 'data/failures_per_time_slot.pkl', 'rb') as f:
        failures_per_time_slot = pickle.load(f)
    with open(base_path + 'data/action_per_step.pkl', 'rb') as f:
        action_per_step = pickle.load(f)
    if mode == 'train':
        with open(base_path + 'data/q_values_per_time_slot.pkl', 'rb') as f:
            q_values_per_time_slot = pickle.load(f)
    else:
        q_values_per_time_slot = []

    rewards_index = find_index(rewards_per_time_slot, epsilon_threshold)
    failures_index = find_index(failures_per_time_slot, epsilon_threshold)
    action_index = find_index(action_per_step, epsilon_threshold)
    q_values_index = find_index(q_values_per_time_slot, epsilon_threshold)

    rewards = [r for r, _ in rewards_per_time_slot[rewards_index:]]
    failures = [f for f, _ in failures_per_time_slot[failures_index:]]
    action_bins = [0]*len(action_bin_labels)
    for action, _ in action_per_step[action_index:]:
        action_bins[action] += 1
    q_values = [q for q, _ in q_values_per_time_slot[q_values_index:]]

    plot_data_online(rewards, idx=1, xlabel='Time Slot', ylabel='Reward',
                     save_path=base_path + 'plots/rewards_per_time_slot.png')
    plot_data_online(failures, idx=2, xlabel='Time Slot', ylabel='Failures',
                     save_path=base_path + 'plots/failures_per_time_slot.png')
    plot_data_online(action_bins, idx=4, xlabel='Action', ylabel='Frequency', show_histogram=True,
                     bin_labels=action_bin_labels, save_path=base_path + 'plots/action_bins.png')
    if len(q_values_per_time_slot) > 0:
        mean_q_values = [np.mean(q) for q in q_values]
        plot_data_online(mean_q_values, idx=3, xlabel='Time Slot', ylabel='Q-Value',
                         save_path=base_path + 'plots/q_values_per_time_slot.png')


def find_index(lst, value):
    return next((i for i, (_, x) in enumerate(lst) if x == value), -1)


if __name__ == '__main__':
    main()