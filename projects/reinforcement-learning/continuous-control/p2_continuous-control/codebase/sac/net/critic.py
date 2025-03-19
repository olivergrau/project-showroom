import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Critic(nn.Module): # QNetwork
    def __init__(self, state_size=33, action_size=4, hidden1=256, hidden2=256):
        super(Critic, self).__init__()
        
        # Q-network 1
        self.fc1_1 = nn.Linear(state_size + action_size, hidden1)
        self.fc1_2 = nn.Linear(hidden1, hidden2)
        self.fc1_3 = nn.Linear(hidden2, 1)
        
        # Q-network 2
        self.fc2_1 = nn.Linear(state_size + action_size, hidden1)
        self.fc2_2 = nn.Linear(hidden1, hidden2)
        self.fc2_3 = nn.Linear(hidden2, 1)
        
        self.apply(weights_init_)
                    
    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)

        # Q1 computation
        x1 = F.relu(self.fc1_1(xu))
        x1 = F.relu(self.fc1_2(x1))
        q1 = self.fc1_3(x1)
        
        # Q2 computation
        x2 = F.relu(self.fc2_1(xu))
        x2 = F.relu(self.fc2_2(x2))
        q2 = self.fc2_3(x2)
        
        return q1, q2
