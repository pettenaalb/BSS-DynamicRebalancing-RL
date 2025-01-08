import torch
import numpy as np
import random

from torch.nn import functional as F

from DuelingDQN import DuelingDQN
from torch_geometric.loader import DataLoader

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for reinforcement learning tasks.

    Attributes:
        - train_model: The Dueling DQN model used for training.
        - target_model: The target Dueling DQN model for stable learning.
        - optimizer: The optimizer for training the train_model.
        - replay_buffer: Experience replay buffer for sampling transitions.
        - num_actions: Number of possible actions in the environment.
        - gamma: Discount factor for future rewards.
        - epsilon: Current value for epsilon-greedy exploration.
        - epsilon_min: Minimum value for epsilon during exploration decay.
        - epsilon_decay: Decay rate for epsilon over time.
        - steps_done: Counter for the number of steps taken, used for epsilon decay.
        - device: Device on which the model is trained (e.g., 'cpu', 'cuda').
    """
    def __init__(self, replay_buffer, num_actions, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500, lr=0.1, device='cpu'):
        """
        Initializes the DQNAgent.

        Parameters:
            - replay_buffer: The replay buffer for experience replay.
            - num_actions: Number of actions available in the environment.
            - gamma: Discount factor for future rewards (default=0.99).
            - epsilon_start: Initial epsilon value for exploration (default=1.0).
            - epsilon_end: Minimum epsilon value after decay (default=0.01).
            - epsilon_decay: Rate of exponential decay for epsilon (default=500).
            - lr: Learning rate for the optimizer (default=0.1).
            - device: Target device for computation (default='cpu').
        """
        self.train_model = DuelingDQN(num_actions).to(device)
        self.target_model = DuelingDQN(num_actions).to(device)
        self.target_model.load_state_dict(self.train_model.state_dict())
        self.optimizer = torch.optim.Adam(self.train_model.parameters(), lr=lr)
        self.replay_buffer = replay_buffer
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.device = device


    def select_action(self, state, greedy=False):
        """
        Selects an action using an epsilon-greedy strategy.

        Parameters:
            - state: The current state of the environment.
            - greedy: If True, selects the greedy action without exploration (default=False).

        Returns:
            - The selected action as an integer.
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            q_values = self.train_model(state)
            return q_values.argmax(dim=-1).item()


    def get_q_values(self, state):
        """
        Returns the Q-values for the given state.

        Parameters:
            - state: The current state of the environment.

        Returns:
            - A tensor of Q-values for each action in the state.
        """
        with torch.no_grad():
            return self.train_model(state)


    def update_epsilon(self, delta_epsilon = None):
        """
        Updates epsilon for the epsilon-greedy strategy using exponential decay.
        """
        if delta_epsilon is None:
            self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        else:
            self.epsilon = max(self.epsilon - delta_epsilon, self.epsilon_min)


    def update_target_network(self):
        """
        Updates the target model by copying the weights from the training model.
        """
        self.target_model.load_state_dict(self.train_model.state_dict())


    def train_step(self, batch_size):
        """
        Performs a single training step using a batch sampled from the replay buffer.

        Parameters:
            - batch_size: The number of samples to draw from the replay buffer.

        Returns:
            - None if the replay buffer does not have enough samples.
        """
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a batch from the replay buffer
        b = self.replay_buffer.sample(batch_size)
        loader = DataLoader(b, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
        batch = next(iter(loader))

        # Compute Q-values for the current states and selected actions, Q(s, a)
        train_q_values = self.train_model(batch, 's').gather(1, batch.actions)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(batch, 't').max(dim=1, keepdim=True)[0]

            # Compute cumulative gamma (sum of discount factors for each step)
            cumulative_discount = self.gamma * (1 - self.gamma ** batch.steps) / (1 - self.gamma)
            discount = self.gamma ** (batch.steps + 1)

            # Final target Q-value equation
            target_q_values = batch.reward + cumulative_discount + discount * next_q_values * (1 - batch.done.float())

        # Compute loss
        loss = F.smooth_l1_loss(train_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.train_model.parameters():
            param.grad.data.clamp_(-1, 1)  # Gradient clipping for stability
        self.optimizer.step()

        # Update epsilon
        # self.steps_done += 1
        # self.update_epsilon()
