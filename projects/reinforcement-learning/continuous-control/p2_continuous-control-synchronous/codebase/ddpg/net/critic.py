import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_size=33, action_size=4, hidden1=400, hidden2=300,
                 use_batch_norm=False, bn_momentum=0.1, init_type='orthogonal',
                 dropout_prob=0.1):
        super(Critic, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.init_type = init_type
        self.dropout_prob = dropout_prob

        self.fc1 = nn.Linear(state_size + action_size, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1, momentum=bn_momentum) if use_batch_norm else None
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2, momentum=bn_momentum) if use_batch_norm else None
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.fc3 = nn.Linear(hidden2, 1)

        self._init_weights()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)  

        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x) 

        return self.fc3(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.fc3:
                    nn.init.uniform_(m.weight, -3e-3, 3e-3)
                    if m.bias is not None:
                        nn.init.uniform_(m.bias, -3e-3, 3e-3)
                else:
                    if self.init_type == 'orthogonal':
                        nn.init.orthogonal_(m.weight, gain=1.0)
                    elif self.init_type == 'kaiming':
                        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    elif self.init_type == 'xavier':
                        nn.init.xavier_uniform_(m.weight)
                    else:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.1)
