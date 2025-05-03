import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor (Policy) Model.

    This class construct the model.
    """

    def __init__(self, state_size, action_size, seed=0, hidden1=256, hidden2=128):
        """ Initialize parameters and build model.

        Args:
            state_size: Integer. Dimension of each state
            action_size: Integer. Dimension of each action
            seed: Integer. Value to set the seed of the model
            fc1_units: Integer. Number of nodes in first fully connect hidden layer
            fc2_units: Integer. Number of nodes in second fully connect hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """ Reset model weights

        :return: None
        """
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    @staticmethod
    def hidden_init(layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return -lim, lim

    def __repr__(self):
        return 'Actor of Deep Deterministic Policy Gradient Model'

    def __str__(self):
        return 'Actor of Deep Deterministic Policy Gradient Model'

    def forward(self, state):
        """Defines the computation performed at every call.

        Args:
            state: A tensor with the state values

        Returns:
            A tensor if there is a single output, or a list of tensors if there
                are more than one outputs.
        """
        # Define the hidden layers
        hidden = F.relu(self.fc1(state))
        hidden = F.relu(self.fc2(hidden))

        return F.tanh(self.fc3(hidden))

# class Actor(nn.Module):
#     def __init__(self, state_size=33, action_size=4, hidden1=256, hidden2=128,
#                  use_batch_norm=False, bn_momentum=0.1, init_type='orthogonal'):
#         super(Actor, self).__init__()
        
#         self.use_batch_norm = use_batch_norm
#         self.init_type = init_type

#         # First hidden layer.
#         self.fc1 = nn.Linear(state_size, hidden1)
#         if self.use_batch_norm:
#             self.bn1 = nn.BatchNorm1d(hidden1, momentum=bn_momentum)
        
#         # Second hidden layer.
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         if self.use_batch_norm:
#             self.bn2 = nn.BatchNorm1d(hidden2, momentum=bn_momentum)
        
#         # Final layer outputs action values.
#         self.fc3 = nn.Linear(hidden2, action_size)
        
#         self._init_weights()
    
#     def forward(self, state):
#         x = self.fc1(state)
        
#         if self.use_batch_norm:
#             x = self.bn1(x)
        
#         x = F.relu(x)
        
#         x = self.fc2(x)
        
#         if self.use_batch_norm:
#             x = self.bn2(x)
        
#         x = F.relu(x)
        
#         # Tanh squashes output between -1 and 1.
#         return torch.tanh(self.fc3(x))
    
#     def _init_weights(self):
#         # Initialize weights for each linear layer.
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 if m == self.fc3:
#                     nn.init.uniform_(m.weight, -3e-3, 3e-3)
#                     if m.bias is not None:
#                         nn.init.uniform_(m.bias, -3e-3, 3e-3)
#                 else:
#                     if self.init_type == 'orthogonal':
#                         nn.init.orthogonal_(m.weight, gain=1.0)
#                     elif self.init_type == 'kaiming':
#                         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                     elif self.init_type == 'xavier':
#                         nn.init.xavier_uniform_(m.weight)
#                     else:
#                         nn.init.normal_(m.weight, mean=0, std=0.1)
#                     if m.bias is not None:
#                         nn.init.constant_(m.bias, 0.1)
