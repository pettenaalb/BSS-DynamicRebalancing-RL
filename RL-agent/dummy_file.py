import platform
import gymnasium as gym
import gymnasium_env.register_env
from tqdm import tqdm
from utils import Actions, plot_data_online

data_path = "../data/"
if platform.system() == "Linux":
    data_path = "/mnt/mydisk/edoardo_scarpel/data/"

# Create the environment
env = gym.make('gymnasium_env/BostonCity-v0', data_path=data_path)
action_size = env.action_space.n

def main():
    options = {
        'total_timeslots': 56,
        'maximum_number_of_bikes': 2500,
    }

    _, _ = env.reset(options=options)

    failures_per_timeslot = []

    not_done = True

    tbar = tqdm(range(options['total_timeslots']), desc='Time Slot', position=0, leave=True)

    total_failures = 0

    while not_done:
        action = Actions.DROP_BIKE.value
        _, _, done, timeslot_terminated, info = env.step(action)
        total_failures += sum(info['failures'])

        # Check if the episode is complete
        not_done = not done

        if timeslot_terminated:
            # Record metrics for the current time slot
            failures_per_timeslot.append(total_failures)
            total_failures = 0
            tbar.update(1)

    print(failures_per_timeslot)
    plot_data_online(failures_per_timeslot, xlabel='Time Slot', ylabel='Failures')



if __name__ == '__main__':
    main()
