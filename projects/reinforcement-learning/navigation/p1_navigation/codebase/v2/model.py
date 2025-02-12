import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Q-Network Model for the Banana environment."""

    def __init__(self, state_size, action_size, seed):
        """
        Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state (37 for the Banana environment)
            action_size (int): Dimension of each action (4 possible actions)
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Increase hidden layer sizes to better capture the 37-dimensional state.
        self.fc1 = nn.Linear(state_size, 128)  # From 37 inputs to 128 nodes.
        self.fc2 = nn.Linear(128, 128)           # Second hidden layer with 128 nodes.
        self.fc3 = nn.Linear(128, action_size)   # Output layer to produce Q-values for 4 actions.

    def forward(self, state):
        """
        Build a network that maps state -> Q-values for each action.
        
        Args:
            state (torch.Tensor): A tensor of shape (batch_size, 37)
        
        Returns:
            torch.Tensor: Q-values for each action, shape (batch_size, 4)
        """
        x = F.relu(self.fc1(state))  # First hidden layer with ReLU activation.
        x = F.relu(self.fc2(x))      # Second hidden layer with ReLU activation.
        q_values = self.fc3(x)       # Output layer (raw Q-values; no activation).
        
        return q_values
