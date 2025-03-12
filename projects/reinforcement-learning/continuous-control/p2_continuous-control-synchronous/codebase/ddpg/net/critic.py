import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_size=33, action_size=4, hidden1=400, hidden2=300):
        super(Critic, self).__init__()

        # The first layer combines state and action.
        self.fc1 = nn.Linear(state_size + action_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)  # Outputs a single Q-value

        # Apply weight initialization here
        self._init_weights()

    def forward(self, state, action):
        # Concatenate state and action along the feature dimension.
        x = torch.cat([state, action], dim=1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.fc3:
                    nn.init.uniform_(m.weight, -3e-3, 3e-3)
                    if m.bias is not None:
                        nn.init.uniform_(m.bias, -3e-3, 3e-3)
                else:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.1)

