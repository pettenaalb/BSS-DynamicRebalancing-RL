import pickle
import numpy as np

from utils import plot_data_online

def main():
    with open('data/rewards_per_time_slot.pkl', 'rb') as f:
        rewards_per_time_slot = pickle.load(f)
    with open('data/failures_per_time_slot.pkl', 'rb') as f:
        failures_per_time_slot = pickle.load(f)
    with open('data/q_values_per_time_slot.pkl', 'rb') as f:
        q_values_per_time_slot = pickle.load(f)
    with open('data/action_bins.pkl', 'rb') as f:
        action_bins = pickle.load(f)

    q_values_len = len(q_values_per_time_slot) // 8
    mean_qvalues_per_timeslot = [np.mean(q_values_per_time_slot[idx * 8:(idx + 1) * 8]) for idx in range(q_values_len)]
    print(mean_qvalues_per_timeslot)

    action_bin_labels = ['STAY', 'RIGHT', 'UP', 'LEFT', 'DOWN', 'DROP_BIKE', 'PICK_UP_BIKE', 'CHARGE_BIKE']

    plot_data_online(rewards_per_time_slot, idx=1, xlabel='Time Slot', ylabel='Reward',
                     save_path='plots/rewards_per_time_slot.png')
    plot_data_online(failures_per_time_slot, idx=2, xlabel='Time Slot', ylabel='Failures',
                     save_path='plots/failures_per_time_slot.png')
    plot_data_online(mean_qvalues_per_timeslot, idx=3, xlabel='Time Slot', ylabel='Q-Value',
                     save_path='plots/q_values_per_time_slot.png')
    plot_data_online(action_bins, idx=4, xlabel='Action', ylabel='Frequency', show_histogram=True,
                     bin_labels=action_bin_labels, save_path='plots/action_bins.png')


if __name__ == '__main__':
    main()