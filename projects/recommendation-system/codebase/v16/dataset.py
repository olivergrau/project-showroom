from torch.utils.data import Dataset
import torch

class CustomMovieLensDataset(Dataset):
    # Static shared properties
    shared_user_profiles = None
    shared_movie_profiles = None
    shared_precomputed_user_histories = None
    shared_precomputed_user_masks = None

    def __init__(self, ratings, genre_vocab, user_token_dims, movie_token_dims, max_history_items=50):
        """
        Args:
            ratings (pd.DataFrame): Must contain ['userIndex', 'movieIndex', 'rating', 'timestamp'] columns.
            genre_vocab (list): List of genre vocabulary strings.
            user_token_dims (dict): e.g. {"age":1, "num_reviews":1, ...}
            movie_token_dims (dict): e.g. {"budget":1, "popularity":1, ...}
            max_history_items (int): Maximum number of historical interactions to keep per user.
        """

        if CustomMovieLensDataset.shared_user_profiles is None:
            raise ValueError("User profiles must be set before creating dataset.")
        
        if CustomMovieLensDataset.shared_movie_profiles is None:
            raise ValueError("Movie profiles must be set before creating dataset.")
        
        if CustomMovieLensDataset.shared_precomputed_user_histories is None:
            raise ValueError("User histories must be set before creating dataset.")
        
        if CustomMovieLensDataset.shared_precomputed_user_masks is None:
            raise ValueError("User masks must be set before creating dataset.")
        
        self.ratings = ratings.copy()
        self.user_indices = ratings['userIndex'].values
        self.movie_indices = ratings['movieIndex'].values

        # Normalize ratings and timestamps
        self.ratings['rating'] = (self.ratings['rating'] - self.ratings['rating'].min()) / (
            self.ratings['rating'].max() - self.ratings['rating'].min()
        )
        self.ratings['timestamp'] = (self.ratings['timestamp'] - self.ratings['timestamp'].min()) / (
            self.ratings['timestamp'].max() - self.ratings['timestamp'].min()
        )
        self.ratings_values = self.ratings['rating'].values

        self.genre_vocab = genre_vocab
        self.user_token_dims = user_token_dims
        self.movie_token_dims = movie_token_dims
        self.max_history_items = max_history_items

    def _extract_genres(self, profile, is_user):
        if is_user:
            genre_part = profile[3 : 3 + len(self.genre_vocab)]
        else:
            genre_part = profile[5 + 384 : 5 + 384 + len(self.genre_vocab)]
        return {
            self.genre_vocab[i]
            for i, val in enumerate(genre_part)
            if val > 0.5
        }

    def _compute_jaccard(self, user_genres, movie_genres):
        if not user_genres or not movie_genres:
            return 0.0
        intersection = len(user_genres & movie_genres)
        union = len(user_genres | movie_genres)
        return intersection / union

    def _slice_tokens(self, full_profile, token_dims):
        profile_dict = {}
        start_idx = 0
        for token_name, dim in token_dims.items():
            end_idx = start_idx + dim
            profile_dict[token_name] = full_profile[start_idx:end_idx]
            start_idx = end_idx
        return profile_dict

    def __getitem__(self, idx):
        """
        Returns a single data point:
            user_tokens, movie_tokens, rating, genre_overlap,
            user_history_tensor, user_history_mask
        """
        user_idx = self.user_indices[idx]
        movie_idx = self.movie_indices[idx]

        # Retrieve shared properties
        user_profile_np = CustomMovieLensDataset.shared_user_profiles[user_idx]  # shape [user_profile_dim]
        movie_profile_np = CustomMovieLensDataset.shared_movie_profiles[movie_idx]  # shape [movie_profile_dim]
        user_history_tensor = CustomMovieLensDataset.shared_precomputed_user_histories[user_idx]
        user_history_mask = CustomMovieLensDataset.shared_precomputed_user_masks[user_idx]

        # Rating and token slicing
        rating = torch.tensor(self.ratings_values[idx], dtype=torch.float32)
        user_tokens = self._slice_tokens(torch.tensor(user_profile_np, dtype=torch.float32), self.user_token_dims)
        movie_tokens = self._slice_tokens(torch.tensor(movie_profile_np, dtype=torch.float32), self.movie_token_dims)

        # Genre overlap
        user_genres = self._extract_genres(user_profile_np, is_user=True)
        movie_genres = self._extract_genres(movie_profile_np, is_user=False)
        genre_overlap = torch.tensor(self._compute_jaccard(user_genres, movie_genres), dtype=torch.float32)

        return (
            user_tokens,
            movie_tokens,
            rating,
            genre_overlap,
            user_history_tensor,
            user_history_mask,
        )

    def __len__(self):
        return len(self.ratings)
