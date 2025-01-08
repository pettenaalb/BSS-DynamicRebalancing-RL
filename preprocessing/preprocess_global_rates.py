import pandas as pd
import numpy as np
import pickle
from utils import kahan_sum
from tqdm import tqdm

days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def main():
    data_path = '../data/'
    global_rates = {}
    tbar = tqdm(total=7*8, desc='Processing global rates')
    for day in days_of_week:
        for timeslot in range(0, 8):
            matrix_path = data_path + 'matrices/09-10/' + str(timeslot).zfill(2) + '/'
            pmf_matrix = pd.read_csv(matrix_path + day.lower() + '-pmf-matrix.csv', index_col='osmid')
            rate_matrix = pd.read_csv(matrix_path + day.lower() + '-rate-matrix.csv', index_col='osmid')

            # Convert index and columns to integers
            pmf_matrix.index = pmf_matrix.index.astype(int)
            pmf_matrix.columns = pmf_matrix.columns.astype(int)
            pmf_matrix.loc[10000, 10000] = 0.0

            global_rate = kahan_sum(rate_matrix.to_numpy().flatten())

            global_rates[(day.lower(), timeslot)] = global_rate

            tbar.update(1)

    print(global_rates)

    with open(data_path + 'utils/global_rates.pkl', 'wb') as f:
        pickle.dump(global_rates, f)

if __name__ == '__main__':
    main()