import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size=33, action_size=4, hidden1=300, hidden2=400):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
    
    def forward(self, state):
        # Assume state is a batch of states with shape [batch_size, 33]
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Tanh squashes the output between -1 and 1, which is typical for continuous actions.
        return torch.tanh(self.fc3(x))
