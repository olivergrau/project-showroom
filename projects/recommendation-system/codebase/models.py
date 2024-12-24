import torch
import torch.nn as nn

class FactorizationMachineLayer(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        """
        Factorization Machine Layer for modeling pairwise interactions.

        Args:
            input_dim (int): Number of input features.
            latent_dim (int): Dimensionality of the latent vectors.
        """
        super(FactorizationMachineLayer, self).__init__()
        self.latent_dim = latent_dim

        # Latent factors for pairwise interactions
        self.v = nn.Parameter(torch.randn(input_dim, latent_dim))

    def forward(self, x):
        """
        Forward pass for the FM layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Interaction term output of shape [batch_size].
        """
        # Compute pairwise interactions using the latent factors
        # (1) Squared sum of embeddings
        squared_sum = torch.pow(torch.matmul(x, self.v), 2)

        # (2) Sum of squared embeddings
        sum_of_squares = torch.matmul(torch.pow(x, 2), torch.pow(self.v, 2))

        # (3) Interaction term
        interactions = 0.5 * torch.sum(squared_sum - sum_of_squares, dim=1)
        return interactions

class RecommendationNetWithFM(nn.Module):
    def __init__(self, user_dim, movie_dim, hidden_dim=128, fm_latent_dim=10, dropout_rate=0.2):
        """
        A neural network for learning user and movie interactions with FM.

        Args:
            user_dim (int): Dimensionality of user profiles.
            movie_dim (int): Dimensionality of movie profiles.
            hidden_dim (int): Number of units in the hidden layers.
            fm_latent_dim (int): Dimensionality of FM latent vectors.
        """
        super(RecommendationNetWithFM, self).__init__()

        # User feature embedding
        self.user_embedding = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Movie feature embedding
        self.movie_embedding = nn.Sequential(
            nn.Linear(movie_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # FM Layer
        self.fm_layer = FactorizationMachineLayer(input_dim=2 * hidden_dim, latent_dim=fm_latent_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, user_profile, movie_profile):
        """
        Forward pass through the network.

        Args:
            user_profile (torch.Tensor): User profile input.
            movie_profile (torch.Tensor): Movie profile input.

        Returns:
            torch.Tensor: Predicted score (e.g., rating).
        """
        # Process user and movie profiles
        user_embed = self.user_embedding(user_profile)
        movie_embed = self.movie_embedding(movie_profile)

        # Concatenate embeddings
        combined = torch.cat([user_embed, movie_embed], dim=1)

        # FM interactions
        fm_interactions = self.fm_layer(combined)

        # Fully connected layers
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)

        # Add FM interactions to the final output
        output = x + fm_interactions.unsqueeze(-1)
        return output, None # compatibility with the training loop

class RecommendationNet(nn.Module):
    def __init__(self, user_dim, movie_dim, hidden_dim=128, dropout_rate=0.2):
        """
        A neural network for learning user and movie interactions with a gating mechanism.
        Args:
            user_dim (int): Dimensionality of user profiles.
            movie_dim (int): Dimensionality of movie profiles.
            hidden_dim (int): Number of units in the hidden layers.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(RecommendationNet, self).__init__()

        # User feature embedding
        self.user_embedding = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Movie feature embedding
        self.movie_embedding = nn.Sequential(
            nn.Linear(movie_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Gating layers for user and movie embeddings
        self.user_gate = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.Sigmoid()  # Gating values between 0 and 1
        )
        self.movie_gate = nn.Sequential(
            nn.Linear(movie_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Combined layers
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, user_profile, movie_profile):
        """
        Forward pass through the network.
        Args:
            user_profile (torch.Tensor): User profile input.
            movie_profile (torch.Tensor): Movie profile input.
        Returns:
            torch.Tensor: Predicted score (e.g., rating).
        """
        # Process user profile with gating mechanism
        user_embed = self.user_embedding(user_profile)
        user_gate = self.user_gate(user_profile)
        gated_user_embed = user_embed * user_gate  # Element-wise multiplication

        # Process movie profile with gating mechanism
        movie_embed = self.movie_embedding(movie_profile)
        movie_gate = self.movie_gate(movie_profile)
        gated_movie_embed = movie_embed * movie_gate  # Element-wise multiplication

        # Concatenate gated embeddings
        combined = torch.cat([gated_user_embed, gated_movie_embed], dim=1)

        # Hidden layers and output
        x = torch.relu(self.fc1(combined))
        output = self.fc2(x)
        return output, None # compatibility with the training loop


class BaselineRecommendationNet(nn.Module):
    def __init__(self, user_dim, movie_dim, hidden_dim=128, dropout_rate=0.2):
        """
        A neural network for learning user and movie interactions.
        Args:
            user_dim (int): Dimensionality of user profiles.
            movie_dim (int): Dimensionality of movie profiles.
            hidden_dim (int): Number of units in the hidden layers.
        """
        super(BaselineRecommendationNet, self).__init__()

        # User feature embedding
        self.user_embedding = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Movie feature embedding
        self.movie_embedding = nn.Sequential(
            nn.Linear(movie_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Combined layers
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, user_profile, movie_profile):
        """
        Forward pass through the network.
        Args:
            user_profile (torch.Tensor): User profile input.
            movie_profile (torch.Tensor): Movie profile input.
        Returns:
            torch.Tensor: Predicted score (e.g., rating).
        """
        # Process user and movie profiles
        user_embed = self.user_embedding(user_profile)
        movie_embed = self.movie_embedding(movie_profile)

        # Concatenate embeddings
        combined = torch.cat([user_embed, movie_embed], dim=1)

        # Hidden layers and output
        x = torch.relu(self.fc1(combined))
        output = self.fc2(x)
        return output, None # compatibility with the training loop
