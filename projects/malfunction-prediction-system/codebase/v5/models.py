import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerRULModel(nn.Module):
    def __init__(
        self, 
        input_dim, 
        d_model=128, 
        nhead=4, 
        num_layers=3, 
        dim_feedforward=256, 
        dropout=0.1
    ):
        """
        Transformer-based model for RUL prediction with Attention-Based Pooling.

        Args:
            input_dim (int): Number of input features per timestep.
            d_model (int): Dimensionality of the model (embedding size).
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dim_feedforward (int): Dimensionality of the feedforward network.
            dropout (float): Dropout rate.
        """
        super().__init__()

        # Input embedding layer to project raw features to d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention-based pooling
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)  # Scalar attention score per timestep
        )

        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Output: single RUL value
        )

    def forward(self, x, mask=None):
        """
        Forward pass for RUL prediction.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            mask (torch.Tensor): Attention mask of shape (batch_size, seq_len), 
                                 where False indicates valid timesteps and True indicates padding.

        Returns:
            torch.Tensor: Predicted RUL of shape (batch_size,).
        """

        # Ensure mask alignment with input
        if mask is not None:
            assert x.size(0) == mask.size(0), "Batch size mismatch between input and mask"
            assert x.size(1) == mask.size(1), "Sequence length mismatch between input and mask"


        # 1. Project input features to d_model
        x = self.input_projection(x)  # Shape: (batch_size, seq_len, d_model)

        # 2. Add positional encodings
        x = self.positional_encoding(x)  # Shape: (batch_size, seq_len, d_model)

        # 3. Pass through Transformer encoder
        # mask must be flipped for Transformer (True = invalid/padded positions)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # Shape: (batch_size, seq_len, d_model)

        # 4. Attention-based pooling
        attn_scores = self.attention(x)  # Shape: (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize over sequence length
        context_vector = (attn_weights * x).sum(dim=1)  # Weighted sum: (batch_size, d_model)

        # 5. Predict RUL using the regressor
        output = self.regressor(context_vector)  # Shape: (batch_size, 1)
        return output.squeeze(1)  # Shape: (batch_size,)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Positional encoding module for adding sequential information to embeddings.

        Args:
            d_model (int): Dimensionality of the model (embedding size).
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        """
        Add positional encodings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encodings added, shape (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)
