import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Q-Network Model for LunarLander-v2."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state (8 for LunarLander-v2)
            action_size (int): Dimension of each action (4 for LunarLander-v2)
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Define a fully connected network with two hidden layers.
        self.fc1 = nn.Linear(state_size, 64)  # First hidden layer: from 8 to 64 nodes
        self.fc2 = nn.Linear(64, 64)          # Second hidden layer: from 64 to 64 nodes
        self.fc3 = nn.Linear(64, action_size) # Output layer: from 64 to 4 nodes (one per action)

    def forward(self, state):
        """Build a network that maps state -> Q-values for each action.
        
        Args:
            state (torch.Tensor): A tensor of shape (batch_size, 8)
        
        Returns:
            torch.Tensor: Q-values for each action, shape (batch_size, 4)
        """
        # Apply the first fully connected layer and a ReLU activation.
        x = F.relu(self.fc1(state))
        
        # Apply the second fully connected layer and a ReLU activation.
        x = F.relu(self.fc2(x))

        # The output layer returns raw Q-values (no activation here).
        q_values = self.fc3(x)
        
        return q_values
