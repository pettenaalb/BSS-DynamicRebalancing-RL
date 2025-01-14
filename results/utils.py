import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np

from matplotlib import pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_data_online(data, show_result=False, idx=1, xlabel='Step', ylabel='Reward', show_histogram=False,
                     bin_labels=None, title="Plot", save_path=None, mean=True):
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
    new_data = []*8
    if isinstance(data, dict):
        for timeslot, value in data.items():
            print(timeslot, value)
            new_data[timeslot] = value
        data = new_data

    # Ensure input data is a NumPy array
    data = np.array(data, dtype=np.float32)

    plt.figure(idx)
    plt.clf()

    if show_histogram:
        plt.title(title)
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
            plt.title(title)
        else:
            plt.title(title)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(data)

        if mean:
            cumulative_mean = np.cumsum(data) / np.arange(1, len(data) + 1, dtype=np.float32)
            plt.plot(cumulative_mean)

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


def plot_confusion_matrix(data: pd.DataFrame, title="Heatmap", x_label = "", y_label = "", cbar_label = "", cmap="YlGnBu", save_path=None):
    """
    Plots a heatmap for failures with days on the x-axis and time slots on the y-axis.

    Parameters:
        - failures: 2D array or DataFrame where rows correspond to time slots and columns to days.
        - days: List of strings for the days of the week (x-axis labels).
        - time_slots: List of strings for the time slots (y-axis labels).
        - title: Title of the heatmap (default: "Failure Heatmap").
        - save_path: Path to save the plot (default: None, which means display it).
    """
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        data=data,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        cbar_kws={'label': cbar_label},
        xticklabels=data.columns,
        yticklabels=data.index
    )

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=15)

    # Set x-axis ticks on top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')  # Move x-axis label to the top

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
