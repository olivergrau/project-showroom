import torch
import torch.nn as nn
import torch.nn.functional as F

# Limits for log standard deviation
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module): # Gaussian Policy
    def __init__(self, state_size=33, action_size=4, hidden1=256, hidden2=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)

        self.mean_linear = nn.Linear(hidden2, action_size)
        self.log_std_linear = nn.Linear(hidden2, action_size)

        self.apply(weights_init_)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_linear(x)
        
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, state):
        """
        Given a state, sample an action using the reparameterization trick.
        Returns:
            - action: The squashed action via tanh, within [-1, 1]
            - log_prob: Log probability of the action (with tanh correction)
            - pre_tanh_value: The raw action values before applying tanh (for diagnostics)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # Reparameterization trick: sample from Normal using rsample()
        pre_tanh = normal.rsample()  # mean + std * noise
        action = torch.tanh(pre_tanh)
        
        # Compute log-probability with tanh correction
        log_prob = normal.log_prob(pre_tanh)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Tanh correction: log(1 - tanh(x)^2)
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1, keepdim=True)
        
        return action, log_prob, pre_tanh
