import torch.nn as nn
import torch_geometric.nn as gnn
import torch

from torch_geometric.nn.conv import GATConv

class DQN(nn.Module):
    def __init__(self, num_actions: int):
        """
        A simple Deep Q-Network with a graph feature extractor (GAT).

        Parameters:
            - num_actions (int): Number of actions in the environment
        """
        super(DQN, self).__init__()

        # Graph feature extractor using Graph Attention Network (GAT)
        self.gat1 = GATConv(in_channels=9, out_channels=64, edge_dim=1)

        # Fully connected layers for graph features
        self.fc_input1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Fully connected layers for agent state features
        self.fc_input2 = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU()
        )

        # Final fully connected layers to predict Q-values
        self.fc_output = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # Initialize weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, batch, key=None):
        """
        Forward pass to compute Q-values.

        Parameters:
            - batch (Data): A batch of graph data.
            - key (str): Key to select source ('s') or target ('t') graph data.

        Returns:
            - q_values (Tensor): Q-values for each action.
        """
        # Extract the appropriate features based on the key
        if key == 's':
            x, edge_index, edge_attr, agent_state, pool_batch = (
                batch.x_s, batch.edge_index_s, batch.edge_attr_s, batch.agent_state, batch.x_s_batch
            )
        elif key == 't':
            x, edge_index, edge_attr, agent_state, pool_batch = (
                batch.x_t, batch.edge_index_t, batch.edge_attr_t, batch.agent_next_state, batch.x_t_batch
            )
        else:
            x, edge_index, edge_attr, agent_state, pool_batch = (
                batch.x, batch.edge_index, batch.edge_attr, batch.agent_state, batch.batch
            )

        # Compute node embeddings using GAT and apply the first fully connected layer
        x = self.gat1(x, edge_index, edge_attr)
        x = self.fc_input1(x)
        x = gnn.global_mean_pool(x, pool_batch)

        # Compute agent state embeddings
        agent_state = self.fc_input2(agent_state)

        # Concatenate graph and agent state embeddings
        try:
            x = torch.cat([x, agent_state], dim=-1)
        except Exception as e:
            print(f"Shape mismatch in concatenation: graph embedding shape {x.shape}, agent state shape {agent_state.shape}")
            raise e

        # Compute Q-values directly
        q_values = self.fc_output(x)
        return q_values