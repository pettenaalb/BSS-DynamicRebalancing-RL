import pickle
import os
import numpy as np

from utils import plot_data_online, plot_dict_data_per_day
from collections import defaultdict

mode = 'train'
action_bin_labels = ['STAY', 'RIGHT', 'UP', 'LEFT', 'DOWN', 'DROP_BIKE', 'PICK_UP_BIKE', 'CHARGE_BIKE']
epsilon_threshold = 0.1

days2num = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
num2days = {0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday', 4: 'friday', 5: 'saturday', 6: 'sunday'}

# ----------------------------------------------------------------------------------------------------------------------

def find_index(lst, eps_thr):
    for i, (_, epsilon) in enumerate(lst):
        if epsilon <= eps_thr:
            return i


def load_data(base_path: str):
    with open(base_path + 'data/rewards_per_timeslot.pkl', 'rb') as f:
        rewards_per_timeslot = pickle.load(f)
    with open(base_path + 'data/failures_per_timeslot.pkl', 'rb') as f:
        failures_per_timeslot = pickle.load(f)
    with open(base_path + 'data/action_per_step.pkl', 'rb') as f:
        action_per_step = pickle.load(f)
    return rewards_per_timeslot, failures_per_timeslot, action_per_step


def process_list(list_data):
    # Initialize a dictionary to hold data per day and timeslot
    data_by_day = defaultdict(lambda: defaultdict(list))

    # Loop over the data and assign values to the corresponding day and timeslot
    for data, timeslot in list_data:
        day_of_week = timeslot // 8  # There are 8 timeslots per day
        daily_slot = timeslot % 8  # Timeslot index within a day (0-7)
        data_by_day[day_of_week][daily_slot].append(data)

    # Compute the mean for each day and timeslot
    result = {}
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for day_index, day_name in enumerate(days):
        result[day_name] = {}
        for slot, values in data_by_day[day_index].items():
            result[day_name][slot] = sum(values) / len(values) if values else 0

    return result

# ----------------------------------------------------------------------------------------------------------------------

def process_training_results(base_path: str):
    rewards_per_timeslot, failures_per_timeslot, action_per_step = load_data(base_path)
    with open(base_path + 'data/q_values_per_timeslot.pkl', 'rb') as f:
        q_values_per_timeslot = pickle.load(f)

    rewards_index = find_index(rewards_per_timeslot, epsilon_threshold)
    failures_index = find_index(failures_per_timeslot, epsilon_threshold)
    action_index = find_index(action_per_step, epsilon_threshold)
    q_values_index = find_index(q_values_per_timeslot, epsilon_threshold)

    epsilons = [e for _, e in rewards_per_timeslot[rewards_index:]]

    rewards = [r for r, _ in rewards_per_timeslot[rewards_index:]]
    failures = [f for f, _ in failures_per_timeslot[failures_index:]]
    action_bins = [0]*len(action_bin_labels)
    for action, _ in action_per_step[action_index:]:
        action_bins[action] += 1
    q_values = [q for q, _ in q_values_per_timeslot[q_values_index:]]

    if not os.path.exists(base_path + 'plots'):
        os.makedirs(base_path + 'plots')
        print(f"Directory '{base_path}plots' created.")

    plot_data_online(rewards, idx=1, xlabel='Time Slot', ylabel='Reward',
                     save_path=base_path + 'plots/rewards_per_timeslot.png')
    plot_data_online(failures, idx=2, xlabel='Time Slot', ylabel='Failures',
                     save_path=base_path + 'plots/failures_per_timeslot.png')
    plot_data_online(action_bins, idx=4, xlabel='Action', ylabel='Frequency', show_histogram=True,
                     bin_labels=action_bin_labels, save_path=base_path + 'plots/action_bins.png')
    mean_q_values = [np.mean(q) for q in q_values]
    plot_data_online(mean_q_values, idx=3, xlabel='Time Slot', ylabel='Q-Value',
                     save_path=base_path + 'plots/q_values_per_timeslot.png')
    plot_data_online(epsilons, idx=5, xlabel='Time Slot', ylabel='Epsilon', save_path=base_path + 'plots/epsilon.png',
                     mean=False)


def process_validation_results(base_path: str):
    rewards_per_timeslot, failures_per_timeslot, action_per_step = load_data(base_path)

    rewards = process_list(rewards_per_timeslot)
    failures = process_list(failures_per_timeslot)

    data_by_day = defaultdict(lambda: defaultdict(list))
    for data, timeslot in action_per_step:
        day_of_week = timeslot // 8
        daily_slot = timeslot % 8
        data_by_day[day_of_week][daily_slot].append(data)
    actions = {}
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for day_index, day_name in enumerate(days):
        actions[day_name] = {}
        for slot, values in data_by_day[day_index].items():
            action_bins = [0]*len(action_bin_labels)
            for value in values:
                action_bins[value] += 1
            actions[day_name][slot] = action_bins

    for day, reward in rewards.items():
        plot_data_online(data=reward.values(), xlabel='Time Slot', ylabel='Reward',
                         save_path=base_path + 'plots/rewards_' + day + '.png')

    for day, failure in failures.items():
        plot_data_online(data=failure.values(), xlabel='Time Slot', ylabel='Failures',
                         save_path=base_path + 'plots/failures_' + day + '.png')


# ----------------------------------------------------------------------------------------------------------------------

def main():
    if mode == 'train':
        base_path = 'training/'
        process_training_results(base_path)
    elif mode == 'validate':
        base_path = 'validation/'
        process_validation_results(base_path)
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'validate'.")


if __name__ == '__main__':
    main()