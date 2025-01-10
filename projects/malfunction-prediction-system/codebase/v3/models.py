import torch
import torch.nn as nn

class RNNBaselineModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout_rate=0.2):
        """
        Initializes the RNN-based baseline model.
        
        Args:
            input_dim (int): Number of input features per time step.
            hidden_dim (int): Number of neurons in the RNN hidden layer.
            num_layers (int): Number of stacked RNN layers.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(RNNBaselineModel, self).__init__()
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Fully connected layer for the last time step
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Forward pass for the RNN model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            torch.Tensor: Predicted RUL of shape (batch_size,).
        """
        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # rnn_out shape: (batch_size, sequence_length, hidden_dim)
        
        # Extract the output corresponding to the last time step
        last_out = rnn_out[:, -1, :]  # shape: (batch_size, hidden_dim)
        
        # Fully connected layer
        output = self.fc(last_out)  # shape: (batch_size, 1)
        
        # Remove extra dimension to match target shape
        return output.squeeze(1)  # shape: (batch_size,)