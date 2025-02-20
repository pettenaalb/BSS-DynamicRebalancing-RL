import torch.nn as nn
import torch_geometric.nn as gnn
import torch
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_mean_pool
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_actions: int):
        """
        A simple Deep Q-Network with a graph feature extractor (GAT).

        Parameters:
            - num_actions (int): Number of actions in the environment
        """
        super(DQN, self).__init__()

        # Graph feature extractor using Graph Attention Network (GAT)
        self.gat1 = GATv2Conv(in_channels=5, out_channels=64, edge_dim=1)

        # Fully connected layers for graph features
        self.fc_input1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )
        # self.fc_input1 = nn.Sequential(
        #     nn.Linear(64, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        # )

        # Fully connected layers for agent state features
        self.fc_input2 = nn.Sequential(
            nn.Linear(60, 64),
            nn.ReLU()
        )
        # self.fc_input2 = nn.Sequential(
        #     nn.Linear(60, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(),
        # )

        # Final fully connected layers to predict Q-values
        self.fc_output = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # --- Graph Path ---
        # self.conv1 = GCNConv(5, 64)
        # self.conv2 = GCNConv(64, 64)
        # self.graph_fc = nn.Linear(64, 64)
        #
        # # --- Truck Path ---
        # self.truck_fc1 = nn.Linear(60, 64)
        # self.truck_fc2 = nn.Linear(64, 64)
        #
        # # --- Fusion & Q-Value Prediction ---
        # self.fc1 = nn.Linear(128, 128)  # Fusion layer
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, num_actions)  # Output Q-values for each action

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

        # x = self.conv1(x, edge_index, edge_attr)
        # x = F.relu(x)
        # x = self.conv2(x, edge_index, edge_attr)
        # x = F.relu(x)
        # x = global_mean_pool(x, pool_batch)
        # x = self.graph_fc(x)
        #
        # # --- Truck Path ---
        # t = F.relu(self.truck_fc1(agent_state))
        # t = F.relu(self.truck_fc2(t))
        #
        # # --- Fusion Layer ---
        # combined = torch.cat([x, t], dim=-1)
        # q = F.relu(self.fc1(combined))
        # q = F.relu(self.fc2(q))
        # q_values = self.fc3(q)

        return q_values