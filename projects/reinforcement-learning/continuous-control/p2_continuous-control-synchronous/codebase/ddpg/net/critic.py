import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """Critic (Value) Model.

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
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1+action_size, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    @staticmethod
    def hidden_init(layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return -lim, lim

    def __repr__(self):
        return 'Critic of Deep Deterministic Policy Gradient Model'

    def __str__(self):
        return 'Critic of Deep Deterministic Policy Gradient Model'

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action)
           pairs -> Q-values.

        Args:
            state: A tensor with the state values
            action: A tensor with the actions values

        Returns:
            A tensor if there is a single output, or a list of tensors if there
                are more than one outputs.
        """
        # Define the hidden layers
        hidden = F.relu(self.fc1(state))
        hidden = torch.cat((hidden, action), dim=1)
        hidden = F.relu(self.fc2(hidden))

        return self.fc3(hidden)


# class Critic(nn.Module):
#     def __init__(self, state_size=33, action_size=4, hidden1=256, hidden2=128,
#                  use_batch_norm=False, bn_momentum=0.1, init_type='orthogonal',
#                  dropout_prob=0.1):
#         super(Critic, self).__init__()

#         self.use_batch_norm = use_batch_norm
#         self.init_type = init_type
#         self.dropout_prob = dropout_prob

#         self.fc1 = nn.Linear(state_size, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1, momentum=bn_momentum) if use_batch_norm else None
        
#         if dropout_prob is not None and dropout_prob > 0:
#             self.dropout1 = nn.Dropout(p=dropout_prob)

#         self.fc2 = nn.Linear(hidden1 + action_size, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2, momentum=bn_momentum) if use_batch_norm else None
        
#         if dropout_prob is not None and dropout_prob > 0:
#             self.dropout2 = nn.Dropout(p=dropout_prob)

#         self.fc3 = nn.Linear(hidden2, 1)

#         self._init_weights()

#     def forward(self, state, action):
#         x = self.fc1(state)
        
#         if self.use_batch_norm:
#             x = self.bn1(x)
        
#         x = F.leaky_relu(x, negative_slope=0.01)

#         if self.dropout_prob is not None and self.dropout_prob > 0:
#             x = self.dropout1(x)  

#         x = torch.cat([x, action], dim=1)

#         x = self.fc2(x)
        
#         if self.use_batch_norm:
#             x = self.bn2(x)
            
#         x = F.leaky_relu(x, negative_slope=0.01)
        
#         if self.dropout_prob is not None and self.dropout_prob > 0:
#             x = self.dropout2(x) 

#         return self.fc3(x)

#     def _init_weights(self):
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
