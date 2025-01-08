import torch
import matplotlib
import numpy as np

from matplotlib import pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_data_online(data, show_result=False, idx=1, xlabel='Step', ylabel='Reward', show_histogram=False,
                     bin_labels=None, save_path=None):
    """
    Plots rewards data online during training or displays final results.

    Parameters:
        - data: List or NumPy array of rewards data.
        - show_result: If True, displays the final results (default=False).
        - idx: Index of the plot figure (default=1).
        - xlabel: Label for the x-axis (default='Step').
        - ylabel: Label for the y-axis (default='Reward').
        - show_histogram: If True, displays a histogram of the data (default=False).
    """
    # Convert input data to a PyTorch tensor
    data = np.array(data)
    data_t = torch.tensor(data, dtype=torch.float)

    plt.figure(idx)
    plt.clf()

    if show_histogram:
        plt.title('Data Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        bins = len(data)
        plt.bar(range(bins), data, alpha=0.75, edgecolor='black')

        # Set custom labels for the x-axis if provided
        if bin_labels is not None:
            if len(bin_labels) != bins:
                raise ValueError("The length of bin_labels must match the number of bins.")
            plt.xticks(ticks=range(bins), labels=bin_labels, rotation=45, ha='right')
    else:
        if show_result:
            plt.title('Result')
        else:
            plt.title('Training...')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(data_t.numpy())

        cumulative_mean = torch.cumsum(data_t, dim=0) / torch.arange(1, len(data_t) + 1, dtype=torch.float32)
        plt.plot(cumulative_mean.numpy())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    plt.close()