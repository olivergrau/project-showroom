import torch
import torch.nn as nn
import time

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
        # Pass through Transformer (batch_first=True, so no transposition needed)
        transformed = self.transformer(features)  # [batch_size, seq_len, d_model]

        # Attention over seq_len dimension
        attn_scores = self.attention_proj(transformed)  # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, seq_len, 1]

        # Weighted sum
        attended_output = torch.sum(transformed * attn_weights, dim=1)  # [batch_size, d_model]
        return attended_output

class TokenEmbedder(nn.Module):
    def __init__(self, token_dims, d_model):
        """
        Parameters:
        -----------
        token_dims : dict
            Dictionary with token names as keys and their input dimensions as values.
            Example: {"age": 1, "genres": 20, "tags": 768}
        d_model : int
            Target embedding size for all tokens.
        """
        super(TokenEmbedder, self).__init__()

        # Combine all token dimensions into a single tensor
        self.token_order = list(token_dims.keys())
        total_input_dim = sum(token_dims.values())

        # Shared linear layer to project all token features
        self.linear = nn.Linear(total_input_dim, len(self.token_order) * d_model)
        self.d_model = d_model

    def forward(self, token_inputs):
        """
        Parameters:
        -----------
        token_inputs : dict
            Dictionary with token names as keys and their corresponding tensors as values.

        Returns:
        --------
        torch.Tensor
            Combined sequence of token embeddings: shape (batch_size, seq_len, d_model)
        """
        # Concatenate all token inputs into a single tensor
        concatenated_inputs = torch.cat([token_inputs[token] for token in self.token_order], dim=1)
        # Shape: (batch_size, total_input_dim)

        # Apply linear projection
        projected = self.linear(concatenated_inputs)
        # Shape: (batch_size, seq_len * d_model)

        # Reshape into sequence format
        batch_size = concatenated_inputs.size(0)
        seq_len = len(self.token_order)
        embedded_tokens = projected.view(batch_size, seq_len, self.d_model)
        # Shape: (batch_size, seq_len, d_model)

        return embedded_tokens

def flatten_tokens_for_fm(token_dict):
    """
    Flatten all token tensors (e.g. age, genres, tags, etc.) along dim=1
    to create a single numeric vector. For example, if token_dict has keys
    ["age": (B,1), "genres": (B,10), "tags": (B,384)], this function returns
    a (B, 1+10+384) tensor.
    """
    return torch.cat([tensor for tensor in token_dict.values()], dim=1)



class RecommendationNet(nn.Module):
    """
    A hybrid content-based recommendation network with optional Transformer and FM blocks,
    and support for multitask learning (e.g., predicting ratings and jaccard similarity for genres).
    """

    def __init__(
        self,
        num_movies,
        genres_vocab_size=0,
        fc_dense_hidden=128,
        fm_latent_dim=16,
        fm_hidden_dim=32,
        transformer_d_model=128,
        transformer_nhead=4,
        transformer_feedforward=256,
        transformer_layers=2,
        activation="ReLU",
        propagate_attention_only=True,
        fm_enabled=True,
        transformer_enabled=True,        
        debug_timings=False,           # <--- Prints timing breakdown if True
    ):
        super(RecommendationNet, self).__init__()

        self.propagate_attention_only = propagate_attention_only
        self.fm_enabled = fm_enabled
        self.transformer_enabled = transformer_enabled
        self.debug_timings = debug_timings

        # Token Dim Dictionaries
        self.user_token_dims = {
            "age": 1, "num_reviews": 1, "avg_rating": 1,
            "genres": genres_vocab_size, "spending_category": 3, "gender": 2, "tags": 384
        }
        self.user_dim = sum(self.user_token_dims.values())

        self.movie_token_dims = {
            "budget": 1, "popularity": 1, "runtime": 1, "vote_average": 1, "vote_count": 1,
            "overview": 384, "genres": genres_vocab_size, "tags": 384
        }
        self.movie_dim = sum(self.movie_token_dims.values())

        # 1. Dense tokenization layers for user and movie profiles
        self.user_embedder = TokenEmbedder(self.user_token_dims, transformer_d_model)
        self.movie_embedder = TokenEmbedder(self.movie_token_dims, transformer_d_model)

        self.activation = self._get_activation(activation)
        self.num_movies = num_movies

        # 2. Factorization Machine (FM)
        if self.fm_enabled:
            self.fm_layer = FactorizationMachineLayer(
                self.user_dim + self.movie_dim, fm_latent_dim
            )
            self.gated_mechanism = GatedMechanism(fm_latent_dim, fm_hidden_dim)
            self.fm_to_token_proj = nn.Linear(fm_hidden_dim, transformer_d_model)
        else:
            self.fm_layer = None
            self.gated_mechanism = None
            self.fm_to_token_proj = None

        # 3. Transformer + Attention
        if self.transformer_enabled:
            self.transformer_block = TransformerAttentionBlock(
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_feedforward,
                num_layers=transformer_layers
            )
        else:
            self.transformer_block = None

        # Calculate how many tokens we place into the sequence
        # Base: user_tokens(7) + movie_tokens(8)
        # 
        # Plus 1 if fm_enabled
        num_sequence_tokens = (
            len(self.user_token_dims)
            + len(self.movie_token_dims)            
            + (1 if self.fm_enabled else 0)
        )

        # Determine dimension for final MLP input
        if self.transformer_enabled:
            if not propagate_attention_only:
                combined_features_dim = (num_sequence_tokens * transformer_d_model) + transformer_d_model
            else:
                combined_features_dim = transformer_d_model
        else:            
            combined_features_dim = transformer_d_model * num_sequence_tokens        

        # 5. Final MLP (prediction head)
        self.fc = nn.Sequential(
            nn.Linear(combined_features_dim, fc_dense_hidden),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(fc_dense_hidden, 1)
        )

        # Print model configuration
        self._print_configuration(
            genres_vocab_size=genres_vocab_size,
            dense_hidden=fc_dense_hidden,
            fm_latent_dim=fm_latent_dim,
            fm_hidden_dim=fm_hidden_dim,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_feedforward=transformer_feedforward,
            transformer_layers=transformer_layers,
            activation=activation,
            propagate_attention_only=propagate_attention_only,
            fm_enabled=fm_enabled,
            transformer_enabled=transformer_enabled,
            debug_timings=debug_timings
        )

    def forward(self, user_tokens, movie_tokens):
        timings = {}
        total_time_start = time.time()

        # 1. Movie Token Embedding
        start_time = time.time()
        movie_emb = self.movie_embedder(movie_tokens)  # Shape: [batch_size, 8, d_model]
        timings["Movie Token Embedding"] = time.time() - start_time

        # 2. User Token Embedding
        start_time = time.time()
        user_emb = self.user_embedder(user_tokens)  # Shape: [batch_size, 7, d_model]
        timings["User Token Embedding"] = time.time() - start_time

        # 4. Optional Factorization Machine
        start_time = time.time()
        fm_token = None
        if self.fm_enabled:
            user_dense = flatten_tokens_for_fm(user_tokens)  # [batch_size, user_dim]
            movie_dense = flatten_tokens_for_fm(movie_tokens)  # [batch_size, movie_dim]
            fm_input = torch.cat([user_dense, movie_dense], dim=1)

            fm_output = self.fm_layer(fm_input)  # [batch_size, fm_latent_dim]
            gated_fm_output = self.gated_mechanism(fm_output)  # [batch_size, fm_hidden_dim]
            fm_token = self.fm_to_token_proj(gated_fm_output).unsqueeze(1)  # [batch_size, 1, d_model]
        timings["Factorization Machine"] = time.time() - start_time

        # 5. Combine tokens for Transformer
        start_time = time.time()
        combined_tokens_list = [user_emb, movie_emb]
        if fm_token is not None:
            combined_tokens_list.append(fm_token)
        
        combined_tokens = torch.cat(combined_tokens_list, dim=1)
        timings["Token Combination"] = time.time() - start_time

        # 6. Optional Transformer + Flatten
        start_time = time.time()
        if self.transformer_enabled:
            attended_features = self.transformer_block(combined_tokens)  # [batch_size, d_model]
            if not self.propagate_attention_only:
                flattened_tokens = combined_tokens.view(combined_tokens.size(0), -1)
                combined_features = torch.cat([flattened_tokens, attended_features], dim=1)
            else:
                combined_features = attended_features
        else:
            combined_features = combined_tokens.view(combined_tokens.size(0), -1)            

        timings["Transformer"] = time.time() - start_time

        # 7. Final MLP
        start_time = time.time()
        rating_pred = self.fc(combined_features)
        timings["Final MLP"] = time.time() - start_time

        total_time = time.time() - total_time_start

        # Print timings only if debug_timings is True
        if self.debug_timings:
            print("\n--- Timing Breakdown ---")
            for stage, t in timings.items():
                pct = (t / total_time) * 100 if total_time > 0 else 0
                print(f"{stage}: {t:.6f} seconds ({pct:.2f}%)")
            print("-------------------------")

        # Return rating and None (for multi-task placeholders)
        return rating_pred, None

    def _get_activation(self, activation):
        if activation == "ReLU":
            return nn.ReLU()
        elif activation == "GeLU":
            return nn.GELU()
        elif activation == "LeakyReLU":
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def _print_configuration(self, **kwargs):
        print("\nModel Configuration:")
        print("=" * 40)
        for key, value in kwargs.items():
            print(f"{key:<30}: {value}")
        print("=" * 40)