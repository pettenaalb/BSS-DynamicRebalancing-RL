import torch
import numpy as np

from torch_geometric.data import Data


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6), device='cpu'):
        self.buffer = []
        self.buffer_size = max_size

        self.device = device

    def push(self, state, action, reward, next_state, done):
        state_graph = state
        next_state_graph = next_state

        transition = PairData(x_s=state_graph.x, edge_index_s=state_graph.edge_index,
                              edge_attr_s=state_graph.edge_attr, edge_type_s=state_graph.edge_type,
                              x_t=next_state_graph.x, edge_index_t=next_state_graph.edge_index,
                              edge_attr_t=next_state_graph.edge_attr, edge_type_t=next_state_graph.edge_type)

        transition.agent_state = torch.tensor(state.agent_state, dtype=torch.float32, device=self.device).unsqueeze(dim=0)
        transition.actions = torch.tensor([action], device=self.device).unsqueeze(dim=1)
        transition.reward = torch.tensor([reward], dtype=torch.float32, device=self.device).unsqueeze(dim=1)
        transition.agent_next_state = torch.tensor(next_state.agent_state, dtype=torch.float32, device=self.device).unsqueeze(dim=0)
        transition.done = torch.tensor([done], device=self.device).unsqueeze(dim=1)
        transition.steps = torch.tensor([state.steps], device=self.device).unsqueeze(dim=1)

        if len(list(self.buffer)) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = []
        for i in ind:
            batch.append(self.buffer[i].to(self.device))

        return batch

    def __len__(self):
        return len(self.buffer)