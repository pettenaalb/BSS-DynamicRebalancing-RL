import torch
import numpy as np
import os
import pickle

from torch_geometric.data import Data


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        """
        Custom increment logic for edge indices to handle source (`_s`) and target (`_t`) data.

        Parameters:
            - key: Key of the attribute.
            - value: Value of the attribute.
            - args: Additional arguments.
            - kwargs: Additional keyword arguments.
        """
        if key == 'edge_index_s':
            return self.x_s.size(0)
        elif key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class PrioritizedReplayBuffer:
    def __init__(self, max_size=int(1e6), alpha=0.6):
        """
        Replay buffer for storing transitions with support for PairData objects.

        Parameters:
            - max_size: Maximum number of transitions to store.
            - device: Device to store the transitions.
        """
        self.buffer = []
        self.buffer_size = max_size
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        """
        Adds a transition to the buffer.

        Parameters:
            - state (PairData): Current state.
            - action (int): Action taken.
            - reward (float): Reward received.
            - next_state (PairData): Next state.
            - done (bool): Whether the episode is complete.
        """
        # Create a PairData transition
        transition = PairData(
            x_s=state.x, edge_index_s=state.edge_index,
            edge_attr_s=state.edge_attr, edge_type_s=state.edge_type,
            x_t=next_state.x, edge_index_t=next_state.edge_index,
            edge_attr_t=next_state.edge_attr, edge_type_t=next_state.edge_type
        )

        # Add additional attributes to the transition
        transition.agent_state = torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(dim=0)
        transition.actions = torch.tensor([action]).unsqueeze(dim=1)
        transition.reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(dim=1)
        transition.agent_next_state = torch.tensor(next_state.agent_state, dtype=torch.float32).unsqueeze(dim=0)
        transition.done = torch.tensor([done]).unsqueeze(dim=1)
        transition.steps = torch.tensor([state.steps]).unsqueeze(dim=1)

        # Compute the priority of the transition
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size, beta=0.4):
        """
        Samples a batch of transitions from the buffer and moves them to the device.

        Parameters:
            - batch_size (int): Number of transitions to sample.

        Returns:
            - List of transitions sampled from the buffer (on the specified device).
        """
        if len(self.buffer) == self.buffer_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[i] for i in indices]

        # Importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        """
        Updates priorities of sampled transitions.

        Parameters:
            - indices: Indices of transitions to update.
            - priorities: New priority values.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        """
        Returns the current size of the buffer.

        Returns:
            - Size of the buffer.
        """
        return len(self.buffer)

    def save_to_files(self, folder_path, chunk_size=10000):
        """
        Saves the replay buffer to multiple files in chunks.

        Parameters:
            - folder_path: Directory to save the buffer files.
            - chunk_size: Number of transitions per file.
        """
        os.makedirs(folder_path, exist_ok=True)
        num_chunks = (len(self.buffer) + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            chunk = self.buffer[i * chunk_size : (i + 1) * chunk_size]
            file_path = os.path.join(folder_path, f"buffer_chunk_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(chunk, f)

    def load_from_files(self, folder_path):
        """
        Loads the replay buffer from files in a folder.

        Parameters:
            - folder_path: Directory containing the buffer files.
        """
        from tqdm import tqdm
        self.buffer = []
        buffer_files = [f for f in os.listdir(folder_path)
                        if f.startswith("buffer_chunk_") and f.endswith(".pkl")]
        buffer_files = sorted(buffer_files)

        tbar = tqdm(total=len(buffer_files), desc="Loading buffer files")
        for file_name in buffer_files:
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "rb") as f:
                    chunk = pickle.load(f)
                    self.buffer.extend(chunk)
            except EOFError:
                print(f"Warning: Skipping file {file_name} due to EOFError.")
            except Exception as e:
                print(f"Error: Failed to load file {file_name} due to {e}.")
            tbar.update(1)