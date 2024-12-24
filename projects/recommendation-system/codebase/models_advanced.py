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
                activation='relu'
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

class AdvancedContentRecommendationNet(nn.Module):
    """
    A hybrid content-based recommendation network that uses:
      - Dense layers for user/movie numeric & multi-hot features
      - A gated mechanism to weight features
      - A Factorization Machine layer for pairwise interactions
      - A Transformer + attention block for advanced feature dependencies
      - A final prediction head (single-task or multi-task)
    """
    def __init__(
        self,
        user_dim,       # dimensionality of user profile (numerical + multi-hot + embeddings)
        movie_dim,      # dimensionality of movie profile (numerical + multi-hot + embeddings)
        dense_hidden=128,
        fm_latent_dim=16,
        transformer_d_model=128,
        transformer_nhead=4,
        transformer_feedforward=256,
        transformer_layers=2,
        multitask=False,   # if True, output rating + genre predictions (example)
        genres_vocab_size=0       # used if multitask is True
    ):
        super(AdvancedContentRecommendationNet, self).__init__()
        self.multitask = multitask

        # 1. Dense layers for user and movie
        #    (reduce dimensionality or transform the raw features)
        self.user_dense = nn.Linear(user_dim, dense_hidden)
        self.movie_dense = nn.Linear(movie_dim, dense_hidden)

        # 2. Gated mechanisms for user and movie
        self.user_gate = GatedMechanism(user_dim, dense_hidden)
        self.movie_gate = GatedMechanism(movie_dim, dense_hidden)

        # 3. Factorization Machine (FM) for pairwise feature interactions
        #    We'll feed the raw features (user_dim + movie_dim) into FM
        self.fm_layer = FactorizationMachineLayer(user_dim + movie_dim, fm_latent_dim)

        # 4. Transformer + Attention for user/movie embeddings
        self.transformer_block = TransformerAttentionBlock(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_feedforward,
            num_layers=transformer_layers
        )

        # Projection to match transformer's d_model
        self.user_transform_proj = nn.Linear(dense_hidden, transformer_d_model)
        self.movie_transform_proj = nn.Linear(dense_hidden, transformer_d_model)

        # 5. Final MLP (prediction head) to produce rating
        #    We'll combine FM output + transform-attended features
        combined_dim = fm_latent_dim + transformer_d_model
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, dense_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dense_hidden, 1)  # single-task rating output
        )

        # If multi-task (e.g., rating + genre prediction),
        # add an additional head for genre classification
        if self.multitask and genres_vocab_size > 0:
            self.genre_head = nn.Linear(combined_dim, genres_vocab_size)
        
        # Print model details
        self.print_model_details(user_dim, movie_dim, dense_hidden, fm_latent_dim, 
                                 transformer_d_model, transformer_nhead, transformer_feedforward,
                                 transformer_layers, multitask, genres_vocab_size)

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
        user_emb_dense = torch.relu(self.user_dense(user_profile))
        movie_emb_dense = torch.relu(self.movie_dense(movie_profile))

        # -------------------------------
        # 2. Gated Mechanisms
        # -------------------------------
        user_gated = self.user_gate(user_profile)   # shape [batch_size, dense_hidden]
        movie_gated = self.movie_gate(movie_profile)

        # Combine user_emb_dense + user_gated (example: sum or concat)
        user_final = user_emb_dense + user_gated
        movie_final = movie_emb_dense + movie_gated

        # -------------------------------
        # 3. Factorization Machine
        #    We feed the raw (user_profile + movie_profile) to FM
        # -------------------------------
        fm_input = torch.cat([user_profile, movie_profile], dim=1)  # [batch_size, user_dim + movie_dim]
        fm_output = self.fm_layer(fm_input)  # [batch_size], the interaction term

        # -------------------------------
        # 4. Transformer + Attention
        #    We'll treat user_final and movie_final as a "sequence" of length 2
        # -------------------------------
        # Project to transformer's d_model
        user_for_transform = self.user_transform_proj(user_final).unsqueeze(1)  # [batch, 1, d_model]
        movie_for_transform = self.movie_transform_proj(movie_final).unsqueeze(1)  # [batch, 1, d_model]

        # Combine into a sequence of length=2
        transform_input = torch.cat([user_for_transform, movie_for_transform], dim=1)  # [batch, 2, d_model]

        # Pass through the transformer + attention block
        attended_features = self.transformer_block(transform_input)  # [batch_size, d_model]

        # -------------------------------
        # 5. Final MLP for rating
        # -------------------------------
        # Combine fm_output + attended_features
        combined = torch.cat([attended_features, fm_output], dim=1)
        # shape is now [batch_size, d_model + 1]

        rating_pred = self.fc(combined)  # [batch_size, 1]

        if not self.multitask:
            return rating_pred, None
        else:
            # Multi-task branch example: genre prediction
            genre_pred = self.genre_head(combined)  # [batch_size, num_genres]
            return rating_pred, genre_pred
    
    def print_model_details(self, user_dim, movie_dim, dense_hidden, fm_latent_dim, 
                            transformer_d_model, transformer_nhead, transformer_feedforward, 
                            transformer_layers, multitask, genres_vocab_size):
        """
        Prints model parameters and architecture details.
        """
        print("Model Details:")
        print(f"  User profile dimension: {user_dim}")
        print(f"  Movie profile dimension: {movie_dim}")
        print(f"  Dense hidden size: {dense_hidden}")
        print(f"  FM latent dimension: {fm_latent_dim}")
        print(f"  Transformer model dimension: {transformer_d_model}")
        print(f"  Transformer attention heads: {transformer_nhead}")
        print(f"  Transformer feedforward dimension: {transformer_feedforward}")
        print(f"  Transformer layers: {transformer_layers}")
        print(f"  Multi-task enabled: {multitask}")
        if multitask:
            print(f"  Genre vocab size: {genres_vocab_size}")
        print("\nModel Summary:")
        print(self)

    def log_metrics(self, epoch, loss, rating_loss=None, genre_loss=None):
        """
        Logs metrics to TensorBoard.

        Args:
            epoch (int): Current training epoch.
            loss (float): Total loss.
            rating_loss (float, optional): Rating prediction loss.
            genre_loss (float, optional): Genre prediction loss.
        """
        self.writer.add_scalar("Loss/Total", loss, epoch)
        if rating_loss is not None:
            self.writer.add_scalar("Loss/Rating", rating_loss, epoch)
        if genre_loss is not None and self.multitask:
            self.writer.add_scalar("Loss/Genre", genre_loss, epoch)