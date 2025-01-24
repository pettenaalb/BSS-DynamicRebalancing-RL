from collections import defaultdict
import pickle

def compute_mean_by_id(data):
    """
    Computes the mean value for each unique ID in the given list of tuples.

    Parameters:
        data (list of tuples): A list where each tuple contains (id, value).

    Returns:
        None: Prints the mean value for each ID.
    """
    # Dictionary to store sums and counts for each ID
    id_sums = defaultdict(float)
    id_counts = defaultdict(int)

    # Iterate through the data
    for id_, value in data:
        id_sums[id_] += value
        id_counts[id_] += 1

    # Compute and print the mean for each ID
    for id_, total in id_sums.items():
        mean_value = total / id_counts[id_]
        print(f"ID: {id_}, Mean Value: {mean_value:.2f} MB")


def main():
    # Sample data
    with open('../data/memory_log.pkl', 'rb') as f:
        data = pickle.load('../data/memory_log.pkl', f)

    # Compute the mean value for each ID
    compute_mean_by_id(data)

if __name__ == "__main__":
    main()