import torch.nn as nn
import torch_geometric.nn as gnn
import torch

from torch_geometric.nn.conv import GATConv

class DuelingDQN(nn.Module):
    def __init__(self, num_actions: int):
        super(DuelingDQN, self).__init__()

        # Graph feature extractor with GAT
        self.gat = GATConv(in_channels=5, out_channels=64, edge_dim=1)

        # Fully connected input layer for graph features
        self.fc_input1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Fully connected input layer for agent state
        self.fc_input2 = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU()
        )

        # Update the first Linear layers to expect 128, because fc_input reduces data to that
        self.value_stream = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Scalar value
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)  # Advantage for each action
        )

    def forward(self, batch, key=None):
        if key == 's':
            x = batch.x_s
            edge_index = batch.edge_index_s
            edge_attr = batch.edge_attr_s
            agent_state = batch.agent_state
            pool_batch = batch.x_s_batch
        elif key == 't':
            x = batch.x_t
            edge_index = batch.edge_index_t
            edge_attr = batch.edge_attr_t
            agent_state = batch.agent_next_state
            pool_batch = batch.x_t_batch
        else:
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            agent_state = batch.agent_state
            pool_batch = batch.batch

        # Compute node embeddings
        x = self.gat(x, edge_index, edge_attr)
        x = self.fc_input1(x)
        x = gnn.global_mean_pool(x, pool_batch)

        # Compute agent state embeddings
        agent_state = self.fc_input2(agent_state)

        # Concatenate node and agent state embeddings
        x = torch.cat([x, agent_state], dim=-1)

        # Compute value and advantage
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage to compute Q-values
        q_values = value + advantage - advantage.mean()
        return q_values