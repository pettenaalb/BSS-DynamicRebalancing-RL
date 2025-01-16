import os
import pandas as pd
import argparse
import pickle

from GPy.util.datasets import data_path

from utils import kahan_sum
from tqdm import tqdm

days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def main(data_path: str):
    if not os.path.exists(data_path + 'utils/'):
        os.makedirs(data_path + 'utils/')
        print('Created directory:', data_path + 'utils/')

    global_rates = {}
    tbar = tqdm(total=7*8, desc='Processing global rates')
    for day in days_of_week:
        for timeslot in range(0, 8):
            matrix_path = data_path + 'matrices/09-10/' + str(timeslot).zfill(2) + '/'
            rate_matrix = pd.read_csv(matrix_path + day.lower() + '-rate-matrix.csv', index_col='osmid')

            global_rate = kahan_sum(rate_matrix.to_numpy().flatten())

            global_rates[(day.lower(), timeslot)] = global_rate

            tbar.update(1)

    print(global_rates)

    with open(data_path + 'utils/global_rates.pkl', 'wb') as f:
        pickle.dump(global_rates, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess global rates')
    parser.add_argument('--data_path', type=str, default='../data/', help='Path to data directory')

    args = parser.parse_args()
    data_path = args.data_path

    main(data_path)