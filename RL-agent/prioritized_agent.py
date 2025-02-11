import torch
import numpy as np
import random

from torch.nn import functional as F

from DuelingDQN import DuelingDQN
from VanillaDQN import DQN
from torch_geometric.loader import DataLoader

class PrioritizedDQNAgent:

    def __init__(self, num_actions, replay_buffer =  None, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=500, lr=0.1, device='cpu', tau=0.005, beta=0.4):
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
        self.train_model = DQN(num_actions).to(device)
        self.target_model = DQN(num_actions).to(device)
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
        self.tau = tau
        self.beta_start = beta
        self.beta = beta


    def select_action(self, state, greedy=False, avoid_action: list = None):
        """
        Selects an action using an epsilon-greedy strategy.

        Parameters:
            - state: The current state of the environment.
            - greedy: If True, selects the greedy action without exploration (default=False).

        Returns:
            - The selected action as an integer.
        """
        if avoid_action is None:
            avoid_action = []

        # Select a random action with probability epsilon
        if not greedy and random.random() < self.epsilon:
            valid_actions = [action for action in range(self.num_actions) if action not in avoid_action]
            if not valid_actions:
                raise ValueError("No valid actions available to select.")

            return random.choice(valid_actions)

        # Select the greedy action
        with torch.no_grad():
            # Get sorted indices of Q-values
            q_values = self.get_q_values(state)
            sorted_q_values = q_values.squeeze(0).detach().argsort(dim=-1, descending=True)

            # Choose the highest-ranked action that is not in avoid_action
            for q_value in sorted_q_values:
                action = q_value.item()
                if action not in avoid_action:
                    del q_values, sorted_q_values
                    return action

        # If cannot find a valid action
        raise ValueError("No valid greedy action could be selected.")

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


    def update_epsilon(self, steps_in_action=1, delta_epsilon = None):
        """
        Updates epsilon for the epsilon-greedy strategy using exponential decay.
        """
        self.steps_done += steps_in_action

        if delta_epsilon is None:
            self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        else:
            self.epsilon = max(self.epsilon - delta_epsilon, self.epsilon_min)


    def update_beta(self):
        """
        Updates beta for prioritized experience replay.
        """
        # Schedule beta
        beta_start = self.beta
        beta_end = 1.0
        self.beta = beta_start + (beta_end - beta_start) * (self.steps_done / self.epsilon_decay)
        self.beta = min(self.beta, beta_end)


    # def update_target_network(self):
    #     """
    #     Updates the target model by copying the weights from the training model.
    #     """
    #     self.target_model.load_state_dict(self.train_model.state_dict())


    def soft_update_target_network(self, tau=0.005):
        for target_param, train_param in zip(self.target_model.parameters(), self.train_model.parameters()):
            target_param.data.copy_(tau * train_param.data + (1 - tau) * target_param.data)


    def train_step(self, batch_size) -> float:
        if len(self.replay_buffer.buffer) < batch_size:
            return 0

        # Sample with priorities and get importance-sampling weights
        batch, indices, weights = self.replay_buffer.sample(batch_size, self.beta)

        loader = DataLoader(batch, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
        batch = next(iter(loader)).to(self.device)

        # Compute Q-values and target Q-values
        train_q_values = self.train_model(batch, 's').gather(1, batch.actions)

        with torch.no_grad():
            next_actions = self.train_model(batch, 't').argmax(dim=1, keepdim=True)
            next_q_values = self.target_model(batch, 't').gather(1, next_actions)

            discount = self.gamma ** batch.steps
            target_q_values = batch.reward + discount * next_q_values * (1 - batch.done.float())

        # Compute TD error and loss
        td_errors = target_q_values - train_q_values
        weights = weights.to(self.device)
        loss = (weights * F.smooth_l1_loss(train_q_values, target_q_values, reduction='none')).mean()

        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), max_norm=10)
        self.optimizer.step()

        # Update priorities in replay buffer based on TD errors
        priorities = td_errors.abs().detach().cpu().numpy() + 1e-5  # Add epsilon to avoid zero priority
        self.replay_buffer.update_priorities(indices, priorities)

        # Perform target network update
        self.soft_update_target_network(tau=self.tau)

        return loss.item()


    def save_model(self, file_path):
        """
        Save the model to a file.

        Parameters:
            - path (str): The path to save the model.
        """
        torch.save(self.train_model.state_dict(), file_path)


    def load_model(self, file_path):
        """
        Load the model from a file.

        Parameters:
            - path (str): The path to load the model.
        """

        self.train_model.load_state_dict(torch.load(file_path, map_location=self.device, weights_only=True))


    def save_checkpoint(self, file_path):
        """
        Save the complete state of the agent for checkpointing.

        Parameters:
            - file_path (str): Path to save the checkpoint.
        """
        checkpoint = {
            'train_model': self.train_model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }
        torch.save(checkpoint, file_path)


    def load_checkpoint(self, file_path):
        """
        Load the state of the agent from a checkpoint.

        Parameters:
            - file_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(file_path, map_location=self.device)
        self.train_model.load_state_dict(checkpoint['train_model'])
        self.target_model.load_state_dict(checkpoint['target_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
