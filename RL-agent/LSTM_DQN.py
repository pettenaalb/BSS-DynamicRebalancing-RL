import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.conv import GATv2Conv

class DQN(nn.Module):
    def __init__(self, num_actions: int, lstm_hidden_dim=64):
        """
        A Deep Q-Network with a Graph Attention Network (GAT) and LSTM for temporal dependencies.

        Parameters:
            - num_actions (int): Number of actions in the environment
            - lstm_hidden_dim (int): Hidden dimension for LSTM
        """
        super(DQN, self).__init__()

        # Graph feature extractor using Graph Attention Network (GAT)
        self.gat1 = GATv2Conv(in_channels=9, out_channels=64, edge_dim=1)

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

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=192, hidden_size=lstm_hidden_dim, batch_first=True)

        # Final fully connected layers to predict Q-values
        self.fc_output = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # Initialize weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, batch, hx=None, key=None):
        """
        Forward pass to compute Q-values.

        Parameters:
            - batch (Data): A batch of graph data.
            - hx (tuple): Hidden and cell state for LSTM (for temporal processing)
            - key (str): Key to select source ('s') or target ('t') graph data.

        Returns:
            - q_values (Tensor): Q-values for each action.
            - hx (tuple): Updated hidden and cell states for LSTM.
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
            x = torch.cat([x, agent_state], dim=-1)  # Shape: [batch_size, 192]
        except Exception as e:
            print(f"Shape mismatch in concatenation: graph embedding shape {x.shape}, agent state shape {agent_state.shape}")
            raise e

        # LSTM processing
        x = x.unsqueeze(1)
        x, hx = self.lstm(x, hx)
        x = x.squeeze(1)

        q_values = self.fc_output(x)
        return q_values, hx