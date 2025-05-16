import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """Weight initialization helper (fan-in)."""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):
    """
    Critic (Q‐value) model with:
      - optional LayerNorm
      - action injected after first hidden layer
      - optional gradient clipping
    """
    def __init__(self,
                 full_obs_size: int,
                 full_action_size: int,
                 hidden_units=[128, 128],
                 seed: int = 0,
                 use_layer_norm: bool = False,    # now controls LayerNorm
                 dropout_p: float = 0.1):
        super(Critic, self).__init__()
        torch.manual_seed(seed)

        self.use_ln = use_layer_norm
        self.dropout_p = dropout_p

        # first layer: obs only
        self.fc1 = nn.Linear(full_obs_size, hidden_units[0])
        if self.use_ln:
            self.ln1 = nn.LayerNorm(hidden_units[0])
        if dropout_p > 0:
            self.dropout1 = nn.Dropout(p=dropout_p)

        # second layer: concat (hidden + actions)
        self.fc2 = nn.Linear(hidden_units[0] + full_action_size, hidden_units[1])
        if self.use_ln:
            self.ln2 = nn.LayerNorm(hidden_units[1])
        if dropout_p > 0:
            self.dropout2 = nn.Dropout(p=dropout_p)

        # output
        self.fc3 = nn.Linear(hidden_units[1], 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self,
                obs_all_agents: torch.Tensor,
                actions_all_agents: torch.Tensor) -> torch.Tensor:
        """
        obs_all_agents: (B, full_obs_size)
        actions_all_agents: (B, full_action_size)
        """
        # 1) obs‐only path
        x = F.relu(self.fc1(obs_all_agents))
        if self.use_ln:
            x = self.ln1(x)
        if self.dropout_p > 0:
            x = self.dropout1(x)

        # 2) inject actions
        x = torch.cat([x, actions_all_agents], dim=-1)

        # 3) joint hidden
        x = F.relu(self.fc2(x))
        if self.use_ln:
            x = self.ln2(x)
        if self.dropout_p > 0:
            x = self.dropout2(x)

        # 4) output Q‐value
        return self.fc3(x)


class Actor(nn.Module):
    """
    Actor (Policy) Model.
    Maps local obs to actions in [-1,1], with optional LayerNorm and no hard clamp.
    """
    def __init__(self,
                 obs_size: int,
                 action_size: int,
                 hidden_units=(128, 128),
                 seed: int = 0,
                 use_layer_norm: bool = False
            ):
        super().__init__()
        torch.manual_seed(seed)

        self.use_ln = use_layer_norm        

        # fully‐connected layers
        h1, h2 = hidden_units
        self.fc1 = nn.Linear(obs_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, action_size)

        if self.use_ln:
            self.ln1 = nn.LayerNorm(h1)
            self.ln2 = nn.LayerNorm(h2)

        self.reset_parameters()

    def reset_parameters(self):
        # same fan‐in init for fc layers
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        # LayerNorm -> weight=1, bias=0
        if self.use_ln:
            self.ln1.weight.data.fill_(1.0)
            self.ln1.bias.data.fill_(0.0)
            self.ln2.weight.data.fill_(1.0)
            self.ln2.bias.data.fill_(0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.fc1(state)
        
        if self.use_ln:
            x = self.ln1(x)
        
        x = F.relu(x)
        x = self.fc2(x)
        
        if self.use_ln:
            x = self.ln2(x)
        
        x = F.relu(x)
        x = torch.tanh(self.fc3(x))          # in [-1,1]

        return x
