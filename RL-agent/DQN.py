import torch.nn.functional as F
from torch import nn


class DeepQNetwork(nn.Module):
    """
    Deep Q-Network (DQN) model for reinforcement learning.

    Attributes:
        - conv_layer_1: First convolutional layer.
        - conv_layer_2: Second convolutional layer.
        - conv_layer_3: Third convolutional layer.
        - dense_layer: Fully connected layer.
        - out_layer: Output layer for Q-values.
    """
    def __init__(self, action_size, hidden_size):
        """
        Initializes the DeepQNetwork.

        Parameters:
            - action_size (int): Number of possible actions in the environment.
            - hidden_size (int): Size of the hidden layer in the network.
        """
        super(DeepQNetwork, self).__init__()
        self.conv_layer_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_layer_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.dense_layer = nn.Linear(7 * 7 * 64, hidden_size)
        self.out_layer = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = x / 255. # image data is stored as ints in 0 to 255 range. Divide to scale to 0 to 1 range
        x = F.relu(self.conv_layer_1(x))
        x = F.relu(self.conv_layer_2(x))
        x = F.relu(self.conv_layer_3(x))
        x = F.relu(self.dense_layer(x.view(x.size(0), -1)))
        return self.out_layer(x)