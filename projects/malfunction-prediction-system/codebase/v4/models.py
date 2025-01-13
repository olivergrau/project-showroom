import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedNet(nn.Module):
    def __init__(
        self,
        input_dim,
        conv_configs=[(64, 3, 1), (128, 3, 1)],  # (out_channels, kernel_size, padding)
        gru_hidden_dim=256,
        fc_hidden_dim=128,
        num_heads=3,
        num_layers=1,
        dropout=0.3,
        attention_heads=4    # Number of attention heads for multi-head self-attention
    ):
        """
        Combined model with parameterizable Convolutional layers, Multi-Head GRU, 
        and a Transformer-style Multi-Head Self-Attention block.

        Args:
            input_dim (int): Number of input features for the first conv layer.
            conv_configs (list of tuples): Each tuple is (out_channels, kernel_size, padding).
            gru_hidden_dim (int): Number of hidden units in each GRU head.
            fc_hidden_dim (int): Number of hidden units in the fully connected layer(s).
            num_heads (int): Number of independent GRU heads.
            num_layers (int): Number of layers in each GRU head.
            dropout (float): Dropout rate for regularization.
            attention_heads (int): Number of parallel attention heads for the MHA block.
        """
        super().__init__()

        # Print model and hyperparameters for quick inspection
        print(f"Model: {self.__class__.__name__}")
        print(f"Input Dimension: {input_dim}")
        print(f"Conv Configs: {conv_configs}")
        print(f"GRU Hidden Dim: {gru_hidden_dim}")
        print(f"FC Hidden Dim: {fc_hidden_dim}")
        print(f"Number of Heads (GRU): {num_heads}")
        print(f"Number of Layers (GRU): {num_layers}")
        print(f"Dropout Rate: {dropout}")
        print(f"Attention Heads (MHA): {attention_heads}")

        self.num_heads = num_heads
        self.gru_hidden_dim = gru_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.attention_heads = attention_heads

        # ------------------
        # 1. Convolutional Feature Processor
        # ------------------
        conv_layers = []
        current_in_channels = input_dim
        for (out_channels, kernel_size, padding) in conv_configs:
            conv_layers.append(
                nn.Conv1d(
                    in_channels=current_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(out_channels))
            current_in_channels = out_channels

        self.feature_processor = nn.Sequential(*conv_layers)
        self.conv_output_dim = current_in_channels

        # ------------------
        # 2. Multi-Head GRU
        # ------------------
        self.gru_heads = nn.ModuleList([
            nn.GRU(
                input_size=self.conv_output_dim,
                hidden_size=self.gru_hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            for _ in range(num_heads)
        ])

        # ------------------
        # 3. Multi-Head Self-Attention (Replacing old single-head attention)
        # ------------------
        # Embed dim for MHA is the total concatenated GRU dimension = gru_hidden_dim * num_heads
        # `attention_heads` is how many parallel attention heads we want in the MHA mechanism.
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.gru_hidden_dim * self.num_heads,
            num_heads=self.attention_heads,
            dropout=dropout,
            batch_first=True  # So shape is (batch, seq, embed_dim)
        )

        # ------------------
        # 4. Fully Connected Layers for Prediction
        # ------------------
        self.fc = nn.Sequential(
            nn.Linear(self.gru_hidden_dim * self.num_heads, self.fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fc_hidden_dim, 1)  # Output: single RUL value
        )

    def forward(self, x):
        """
        Forward pass for the combined model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Predicted RUL of shape (batch_size,).
        """
        # 1. Feature Processing via Convolution
        #    We expect x to be (batch_size, seq_len, input_dim).
        #    For Conv1d, we need (batch_size, input_dim, seq_len).
        x = x.permute(0, 2, 1)
        x = self.feature_processor(x)
        # Now shape is (batch_size, conv_output_dim, seq_len).
        # Convert back to (batch_size, seq_len, conv_output_dim).
        x = x.permute(0, 2, 1)

        # 2. Multi-Head GRU
        head_outputs = []
        for head in self.gru_heads:
            # gru_out shape: (batch_size, seq_len, gru_hidden_dim)
            gru_out, _ = head(x)
            head_outputs.append(gru_out)

        # Concatenate GRU outputs along the feature dimension
        # shape: (batch_size, seq_len, gru_hidden_dim * num_heads)
        combined = torch.cat(head_outputs, dim=-1)

        # 3. Multi-Head Self-Attention
        # The MHA expects (batch_size, seq_len, embed_dim) for all three inputs
        # We'll treat `combined` as query, key, and value (self-attention).
        attn_out, attn_weights = self.multihead_attention(
            combined,  # query
            combined,  # key
            combined   # value
        )
        # attn_out shape: (batch_size, seq_len, embed_dim)
        # attn_weights shape: (batch_size, seq_len, seq_len)

        # Pool across the sequence to get a single context vector per batch element.
        # You could use mean, sum, or even take the embedding corresponding to a CLS token if you had one.
        context_vector = attn_out.mean(dim=1)  # shape: (batch_size, embed_dim)

        # 4. Final Prediction
        output = self.fc(context_vector)  # shape: (batch_size, 1)
        return output.squeeze(1)

class MultiHeadGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=3, num_layers=1, dropout=0.3):
        """
        Multi-Head GRU-based model for RUL prediction.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in each GRU head.
            num_heads (int): Number of independent GRU heads.
            num_layers (int): Number of layers in each GRU head.
            dropout (float): Dropout rate applied to each GRU head.
        """
        super(MultiHeadGRU, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Create GRU heads
        self.gru_heads = nn.ModuleList([
            nn.GRU(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0.0
            ) for _ in range(num_heads)
        ])

        # Fully Connected Layers for Combining GRU Outputs
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Output: single RUL value
        )
    
    def forward(self, x):
        """
        Forward pass for the Multi-Head GRU-based model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Predicted RUL of shape (batch_size,).
        """
        # Process input through each GRU head
        head_outputs = []
        for head in self.gru_heads:
            gru_out, _ = head(x)  # Shape: (batch_size, sequence_length, hidden_dim)
            last_hidden_state = gru_out[:, -1, :]  # Use the last hidden state (batch_size, hidden_dim)
            head_outputs.append(last_hidden_state)

        # Concatenate outputs from all heads
        combined = torch.cat(head_outputs, dim=-1)  # Shape: (batch_size, hidden_dim * num_heads)

        # Fully connected layers for prediction
        output = self.fc(combined)  # Shape: (batch_size, 1)
        return output.squeeze(1)  # Output shape: (batch_size,)

class FeatureProcessingGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3, use_residual=False):
        """
        Enhanced GRU-based model with feature processing and optional residual connections.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in each GRU layer.
            num_layers (int): Number of stacked GRU layers.
            dropout (float): Dropout rate for regularization.
            use_residual (bool): Whether to use residual connections in the GRU.
        """
        super(FeatureProcessingGRU, self).__init__()
        
        self.use_residual = use_residual

        # Feature Processing Layers
        self.feature_processor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # GRU Layers
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Output: single RUL value
        )
    
    def forward(self, x):
        """
        Forward pass for the enhanced GRU model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Predicted RUL of shape (batch_size,).
        """
        # Feature Processing
        # Convert shape (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        x = self.feature_processor(x)  # Shape: (batch_size, 128, seq_len)
        x = x.permute(0, 2, 1)  # Convert back to (batch_size, seq_len, 128)

        # GRU forward pass
        gru_out, _ = self.gru(x)  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Optional Residual Connection
        if self.use_residual:
            residual = x[:, -gru_out.size(1):, :]  # Match sequence length for residual
            gru_out += residual  # Add residual connection

        # Use the last time step's output
        last_hidden_state = gru_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        
        # Fully connected layers for prediction
        output = self.fc(last_hidden_state)  # Shape: (batch_size, 1)
        return output.squeeze(1)  # Output shape: (batch_size,)


class GRUWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        GRU-based model with an attention mechanism for RUL prediction.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in each GRU layer.
            num_layers (int): Number of stacked GRU layers.
            dropout (float): Dropout rate for regularization.
        """
        super(GRUWithAttention, self).__init__()
        
        # GRU layers
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Project hidden state
            nn.Tanh(),  # Activation for attention scores
            nn.Linear(hidden_dim, 1, bias=False)  # Compute scalar attention scores
        )
        
        # Fully connected layers for prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Output: single RUL value
        )
    
    def forward(self, x):
        """
        Forward pass with attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Predicted RUL of shape (batch_size,).
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Attention mechanism
        attention_scores = self.attention(gru_out)  # Shape: (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # Normalize scores (batch_size, seq_len, 1)
        context_vector = torch.sum(gru_out * attention_weights, dim=1)  # Weighted sum (batch_size, hidden_dim)
        
        # Fully connected layers
        output = self.fc(context_vector)  # Shape: (batch_size, 1)
        return output.squeeze(1)  # Output shape: (batch_size,)

class GRUBasedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        A GRU-based model for processing sequences and predicting RUL.

        Args:
            input_dim (int): Number of input features (feature_num).
            hidden_dim (int): Number of hidden units in each GRU layer.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout rate applied between GRU layers.
        """
        super(GRUBasedModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU Layers
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # Reduce dimensionality after GRU
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Output layer (single value for RUL)
        )
        
    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            torch.Tensor: Predicted RUL of shape (batch_size,).
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)  # gru_out: (batch_size, sequence_length, hidden_dim)
        
        # Use only the last time step's output
        last_hidden_state = gru_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Fully connected layers for prediction
        output = self.fc(last_hidden_state)  # (batch_size, 1)
        return output.squeeze(1)  # Output shape: (batch_size,)

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