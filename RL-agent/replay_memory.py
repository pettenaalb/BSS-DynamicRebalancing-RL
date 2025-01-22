import torch
import numpy as np

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


class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        """
        Replay buffer for storing transitions with support for PairData objects.

        Parameters:
            - max_size: Maximum number of transitions to store.
            - device: Device to store the transitions.
        """
        self.buffer = []
        self.buffer_size = max_size

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

        # Maintain buffer size by removing oldest transition if full
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the buffer and moves them to the device.

        Parameters:
            - batch_size (int): Number of transitions to sample.

        Returns:
            - List of transitions sampled from the buffer (on the specified device).
        """
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]

        return batch

    def __len__(self):
        """
        Returns the current size of the buffer.

        Returns:
            - Size of the buffer.
        """
        return len(self.buffer)