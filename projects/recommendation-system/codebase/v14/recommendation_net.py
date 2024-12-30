import torch
import torch.nn as nn

class FactorizationMachineLayer(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        """
        Factorization Machine Layer for modeling pairwise interactions.
        
        Args:
            input_dim (int): Number of input features.
            latent_dim (int): Dimensionality of latent vectors for feature interactions.
        """
        super(FactorizationMachineLayer, self).__init__()
        self.latent_dim = latent_dim

        # Latent factors for pairwise interactions
        self.v = nn.Parameter(torch.randn(input_dim, latent_dim))

    def forward(self, x):
        """
        Forward pass for the FM layer.
        
        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, input_dim].

        Returns:
            torch.Tensor: FM interaction output, shape [batch_size, latent_dim].
        """
        # (1) Squared sum of embeddings
        squared_sum = torch.pow(torch.matmul(x, self.v), 2)  # [batch_size, latent_dim]

        # (2) Sum of squared embeddings
        sum_of_squares = torch.matmul(torch.pow(x, 2), torch.pow(self.v, 2))  # [batch_size, latent_dim]

        # (3) Vector of pairwise interactions
        interactions = 0.5 * (squared_sum - sum_of_squares)  # [batch_size, latent_dim]
        return interactions

class GatedMechanism(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedMechanism, self).__init__()
        # Project to output_dim (can match the dimension of the dense embedding)
        self.gate = nn.Linear(input_dim, output_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        x shape: [batch_size, input_dim]
        """
        # Gating values (between 0 and 1)
        g = torch.sigmoid(self.gate(x))  # [batch_size, output_dim]
        # Projection
        h = torch.relu(self.proj(x))     # [batch_size, output_dim]
        # Element-wise scaling by gate
        return g * h

class TransformerAttentionBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, num_layers=2):
        super(TransformerAttentionBlock, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation='relu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        # Simple linear attention
        self.attention_proj = nn.Linear(d_model, 1)

    def forward(self, features):
        """
        Args:
            features (torch.Tensor): [batch_size, seq_len, d_model]
                Here seq_len can be 2 (user, movie) or more if you treat features as a sequence.

        Returns:
            torch.Tensor: attention-weighted feature, shape [batch_size, d_model].
        """
        # Prepare for transformer: shape => [seq_len, batch_size, d_model]
        features_t = features.transpose(0, 1)  # [seq_len, batch_size, d_model]

        # Pass through Transformer
        transformed = self.transformer(features_t)  # [seq_len, batch_size, d_model]

        # Revert to [batch_size, seq_len, d_model]
        transformed = transformed.transpose(0, 1)

        # Attention over seq_len dimension
        # shape = [batch_size, seq_len, 1]
        attn_scores = self.attention_proj(transformed)
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, seq_len, 1]

        # Weighted sum
        # [batch_size, seq_len, d_model] * [batch_size, seq_len, 1] => [batch_size, d_model]
        attended_output = torch.sum(transformed * attn_weights, dim=1)
        return attended_output

class ConfigurableRecommendationNet(nn.Module):
    """
    A hybrid content-based recommendation network with optional Transformer and FM blocks,
    and support for multitask learning (e.g., predicting ratings and auxiliary labels).
    """
    def __init__(
        self,
        user_dim,       # Dimensionality of user profile (numerical + multi-hot + embeddings)
        movie_dim,      # Dimensionality of movie profile (numerical + multi-hot + embeddings)
        use_transformer=True,  # Toggle Transformer block
        use_fm=True,           # Toggle Factorization Machine block
        multitask=False,       # Enable multitask learning
        genres_vocab_size=0,   # Vocabulary size for auxiliary label prediction
        dense_hidden=128, # Hidden size for dense layers
        fm_latent_dim=16, # Latent dimension for FM layer
        fm_hidden_dim=32, # Hidden size for FM projection
        transformer_d_model=128, # Model dimension for Transformer
        transformer_nhead=4, # Attention heads for Transformer
        transformer_feedforward=256, # Feedforward dimension for Transformer
        transformer_layers=2,   # Number of layers for Transformer
        activation="ReLU"  # Activation function: "ReLU", "GeLU", or "LeakyReLU"
    ):
        super(ConfigurableRecommendationNet, self).__init__()

        self.use_transformer = use_transformer
        self.use_fm = use_fm
        self.multitask = multitask
        self.activation = self._get_activation(activation)

        # 1. Dense layers for user and movie
        self.user_dense = nn.Linear(user_dim, dense_hidden)
        self.movie_dense = nn.Linear(movie_dim, dense_hidden)

        # 2. Optional Factorization Machine (FM) for pairwise feature interactions
        if self.use_fm:
            self.fm_layer = FactorizationMachineLayer(user_dim + movie_dim, fm_latent_dim)            
            self.fm_proj = nn.Sequential(
                nn.Linear(fm_latent_dim, fm_hidden_dim),
                self.activation,
                nn.Dropout(0.2)
            )

        # 3. Optional Transformer + Attention for user/movie embeddings
        if self.use_transformer:
            self.transformer_block = TransformerAttentionBlock(
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_feedforward,
                num_layers=transformer_layers
            )
            self.user_transform_proj = nn.Linear(dense_hidden, transformer_d_model)
            self.movie_transform_proj = nn.Linear(dense_hidden, transformer_d_model)

        # 4. Final MLP (prediction head)
        combined_dim = dense_hidden
        if self.use_transformer:
            combined_dim += transformer_d_model
        if self.use_fm:
            combined_dim += fm_hidden_dim

        # Single-task prediction (rating prediction)
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, dense_hidden),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(dense_hidden, 1)  # Single-task rating output
        )

        # Optional multitask prediction (e.g., genre classification)
        if self.multitask and genres_vocab_size > 0:
            self.genre_head = nn.Linear(combined_dim, genres_vocab_size)

        # Print model configuration
        self._print_configuration(
            user_dim=user_dim,
            movie_dim=movie_dim,
            use_transformer=use_transformer,
            use_fm=use_fm,
            multitask=multitask,
            genres_vocab_size=genres_vocab_size,
            dense_hidden=dense_hidden,
            fm_latent_dim=fm_latent_dim,
            fm_hidden_dim=fm_hidden_dim,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_feedforward=transformer_feedforward,
            transformer_layers=transformer_layers,
            activation=activation
        )

    def forward(self, user_profile, movie_profile):
        """
        Args:
            user_profile (torch.Tensor): [batch_size, user_dim]
            movie_profile (torch.Tensor): [batch_size, movie_dim]

        Returns:
            If multitask=False:
                rating_pred (torch.Tensor): [batch_size, 1]
            If multitask=True:
                (rating_pred, genre_pred)
                  rating_pred: [batch_size, 1]
                  genre_pred: [batch_size, num_genres]
        """
        # -------------------------------
        # 1. Dense transformations
        # -------------------------------
        user_emb_dense = self.activation(self.user_dense(user_profile))
        movie_emb_dense = self.activation(self.movie_dense(movie_profile))

        # Combine user and movie embeddings
        combined_features = user_emb_dense + movie_emb_dense

        # -------------------------------
        # 2. Optional Factorization Machine
        # -------------------------------
        if self.use_fm:
            fm_input = torch.cat([user_profile, movie_profile], dim=1)
            fm_output = self.fm_layer(fm_input)  # [batch_size, fm_latent_dim]
            fm_output = self.fm_proj(fm_output)  # Non-linear transformation
            combined_features = torch.cat([combined_features, fm_output], dim=1)

        # -------------------------------
        # 3. Optional Transformer + Attention
        # -------------------------------
        if self.use_transformer:
            user_transformed = self.user_transform_proj(user_emb_dense).unsqueeze(1)
            movie_transformed = self.movie_transform_proj(movie_emb_dense).unsqueeze(1)
            transform_input = torch.cat([user_transformed, movie_transformed], dim=1)
            attended_features = self.transformer_block(transform_input)
            combined_features = torch.cat([combined_features, attended_features], dim=1)

        # -------------------------------
        # 4. Final MLP for rating
        # -------------------------------
        rating_pred = self.fc(combined_features)

        if not self.multitask:
            return rating_pred, None
        else:
            # Multi-task branch example: genre prediction
            genre_pred = self.genre_head(combined_features)  # [batch_size, num_genres]
            return rating_pred, genre_pred

    def _get_activation(self, activation):
        """
        Helper method to return the activation function based on the parameter.
        """
        if activation == "ReLU":
            return nn.ReLU()
        elif activation == "GeLU":
            return nn.GELU()
        elif activation == "LeakyReLU":
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def _print_configuration(self, **kwargs):
        """
        Print the model configuration in a tabular format.
        """
        print("\nModel Configuration:")
        print("=" * 40)
        for key, value in kwargs.items():
            print(f"{key:<25}: {value}")
        print("=" * 40)

class RecommendationNet(nn.Module):
    """
    A hybrid content-based recommendation network with optional Transformer and FM blocks,
    and support for multitask learning (e.g., predicting ratings and auxiliary labels).
    """
    def __init__(
        self,
        user_dim,       # Dimensionality of user profile (numerical + multi-hot + embeddings)
        movie_dim,      # Dimensionality of movie profile (numerical + multi-hot + embeddings)
        use_transformer=True,  # Toggle Transformer block
        use_fm=True,           # Toggle Factorization Machine block
        multitask=False,       # Enable multitask learning
        genres_vocab_size=0,   # Vocabulary size for auxiliary label prediction
        dense_hidden=128, # Hidden size for dense layers
        fm_latent_dim=16, # Latent dimension for FM layer
        fm_hidden_dim=32, # Hidden size for FM projection
        transformer_d_model=128, # Model dimension for Transformer
        transformer_nhead=4, # Attention heads for Transformer
        transformer_feedforward=256, # Feedforward dimension for Transformer
        transformer_layers=2,   # Number of layers for Transformer
        activation="ReLU"  # Activation function: "ReLU", "GeLU", or "LeakyReLU"
    ):
        super(RecommendationNet, self).__init__()

        self.use_transformer = use_transformer
        self.use_fm = use_fm
        self.multitask = multitask
        self.activation = self._get_activation(activation)

        # 1. Dense layers for user and movie
        self.user_dense = nn.Linear(user_dim, dense_hidden)
        self.movie_dense = nn.Linear(movie_dim, dense_hidden)

        # 2. Optional Factorization Machine (FM) for pairwise feature interactions
        if self.use_fm:
            self.fm_layer = FactorizationMachineLayer(user_dim + movie_dim, fm_latent_dim)            
            self.fm_proj = nn.Sequential(
                nn.Linear(fm_latent_dim, fm_hidden_dim),
                self.activation,
                nn.Dropout(0.2)
            )

        # 3. Optional Transformer + Attention for user/movie embeddings
        if self.use_transformer:
            self.transformer_block = TransformerAttentionBlock(
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_feedforward,
                num_layers=transformer_layers
            )
            self.user_transform_proj = nn.Linear(dense_hidden, transformer_d_model)
            self.movie_transform_proj = nn.Linear(dense_hidden, transformer_d_model)

        # 4. Final MLP (prediction head)
        combined_dim = dense_hidden
        if self.use_transformer:
            combined_dim += transformer_d_model
        if self.use_fm:
            combined_dim += fm_hidden_dim

        # Single-task prediction (rating prediction)
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, dense_hidden),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(dense_hidden, 1)  # Single-task rating output
        )

        # Optional multitask prediction (predicting Jaccard similarity instead of classification)
        if self.multitask:
            self.jaccard_head = nn.Sequential(
                nn.Linear(combined_dim, dense_hidden),
                self.activation,
                nn.Dropout(0.2),
                nn.Linear(dense_hidden, 1),  # Jaccard similarity output (single scalar)
                nn.Sigmoid()  # Ensures output is in [0, 1]
            )

        # Print model configuration
        self._print_configuration(
            user_dim=user_dim,
            movie_dim=movie_dim,
            use_transformer=use_transformer,
            use_fm=use_fm,
            multitask=multitask,
            genres_vocab_size=genres_vocab_size,
            dense_hidden=dense_hidden,
            fm_latent_dim=fm_latent_dim,
            fm_hidden_dim=fm_hidden_dim,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_feedforward=transformer_feedforward,
            transformer_layers=transformer_layers,
            activation=activation
        )

    def forward(self, user_profile, movie_profile):
        """
        Args:
            user_profile (torch.Tensor): [batch_size, user_dim]
            movie_profile (torch.Tensor): [batch_size, movie_dim]

        Returns:
            If multitask=False:
                rating_pred (torch.Tensor): [batch_size, 1]
            If multitask=True:
                (rating_pred, jaccard_pred)
                  rating_pred: [batch_size, 1]
                  jaccard_pred: [batch_size, 1]
        """
        # -------------------------------
        # 1. Dense transformations
        # -------------------------------
        user_emb_dense = self.activation(self.user_dense(user_profile))
        movie_emb_dense = self.activation(self.movie_dense(movie_profile))

        # Combine user and movie embeddings
        combined_features = user_emb_dense + movie_emb_dense

        # -------------------------------
        # 2. Optional Factorization Machine
        # -------------------------------
        if self.use_fm:
            fm_input = torch.cat([user_profile, movie_profile], dim=1)
            fm_output = self.fm_layer(fm_input)  # [batch_size, fm_latent_dim]
            fm_output = self.fm_proj(fm_output)  # Non-linear transformation
            combined_features = torch.cat([combined_features, fm_output], dim=1)

        # -------------------------------
        # 3. Optional Transformer + Attention
        # -------------------------------
        if self.use_transformer:
            user_transformed = self.user_transform_proj(user_emb_dense).unsqueeze(1)
            movie_transformed = self.movie_transform_proj(movie_emb_dense).unsqueeze(1)
            transform_input = torch.cat([user_transformed, movie_transformed], dim=1)
            attended_features = self.transformer_block(transform_input)
            combined_features = torch.cat([combined_features, attended_features], dim=1)

        # -------------------------------
        # 4. Final MLP for rating
        # -------------------------------
        rating_pred = self.fc(combined_features)

        if not self.multitask:
            return rating_pred, None
        else:
            # Multi-task branch for predicting Jaccard similarity
            jaccard_pred = self.jaccard_head(combined_features)  # [batch_size, 1]
            return rating_pred, jaccard_pred

    def _get_activation(self, activation):
        """
        Helper method to return the activation function based on the parameter.
        """
        if activation == "ReLU":
            return nn.ReLU()
        elif activation == "GeLU":
            return nn.GELU()
        elif activation == "LeakyReLU":
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def _print_configuration(self, **kwargs):
        """
        Print the model configuration in a tabular format.
        """
        print("\nModel Configuration:")
        print("=" * 40)
        for key, value in kwargs.items():
            print(f"{key:<25}: {value}")
        print("=" * 40)
