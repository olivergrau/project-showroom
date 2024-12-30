# ----------------------------------------------------------------
# Profile Creator Interfaces
# ----------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
import torch
from torch.utils.data import Dataset

class IProfileCreator:
    """Interface for profile creators."""

    def create_profiles(self, df, *args, **kwargs):
        """Generate numeric embeddings (profiles) from a DataFrame."""
        raise NotImplementedError

# ----------------------------------------------------------------
# Movie Profile Creator
# ----------------------------------------------------------------
class MovieProfileCreator(IProfileCreator):
    """
    Responsible for creating movie profiles by combining:
      - Normalized numeric features (budget, popularity, runtime, vote_average, vote_count)
      - Overview embeddings (batch-encoded with text model)
      - Multi-hot genres
      - Tag embeddings (batch-encoded with text model)
    """

    def __init__(self, text_model_tags, text_model_plots, genre_vocab, numeric_cols=None):
        """
        Parameters:
        -----------
        text_model : SentenceTransformer or similar
            Model to encode textual data
        genre_vocab : list
            All unique genres extracted from user/movie data
        numeric_cols : list
            Numeric columns to normalize for movies
        """
        self.text_model_tags = text_model_tags
        self.text_model_plots = text_model_plots
        self.genre_vocab = genre_vocab
        self.numeric_cols = numeric_cols or [
            'budget', 'popularity', 'runtime', 'vote_average', 'vote_count'
        ]

    def create_profiles(self, movies_df):
        """Main entry point for creating the movie profile matrix."""

        # 1) Multi-hot genres
        movie_genres_encoded = multi_hot_encode_genres(
            movies_df,
            self.genre_vocab,
            split='|',
            genres_column='genres'
        )       

        # 2) Text embeddings: Overviews
        overview_list = movies_df['overview'].fillna("").tolist()
        movie_overview_embeddings = batch_encode_text(
            overview_list, self.text_model_plots, convert_to_tensor=True
        )

        # 3) Text embeddings: Tags
        movies_df['tags_by_users'] = movies_df['tags_by_users'].fillna("")
        tags_list = movies_df['tags_by_users'].tolist()
        movie_tag_embeddings = batch_encode_text(
            tags_list, self.text_model_tags, convert_to_tensor=True
        )

        # 4) Normalize numeric columns
        movies_df[self.numeric_cols] = movies_df[self.numeric_cols].fillna(0)
        movies_df = normalize_multiple_columns(movies_df, self.numeric_cols)
        numeric_features = movies_df[self.numeric_cols].to_numpy()  # shape: [num_movies, 5]

        # 5) Combine all features
        movie_profiles = np.hstack([
            numeric_features,          # shape: [num_movies, 5]
            movie_overview_embeddings, # shape: [num_movies, overview_dim]
            movie_genres_encoded,      # shape: [num_movies, len(genre_vocab)]
            movie_tag_embeddings       # shape: [num_movies, tag_dim]
        ])

        return movie_profiles


# ----------------------------------------------------------------
# User Profile Creator
# ----------------------------------------------------------------
class UserProfileCreator(IProfileCreator):
    """
    Responsible for creating user profiles by combining:
      - Normalized numeric features (e.g., age)
      - Multi-hot favorite genres
      - Tag embeddings
      - one-hot sex
    """

    def __init__(self, text_model, genre_vocab, numeric_cols=['age', 'num_reviews', 'avg_rating']):
        """
        Parameters:
        -----------
        text_model : SentenceTransformer or similar
            Model to encode textual data
        genre_vocab : list
            All unique genres extracted from user/movie data
        numeric_cols : list
            Numeric columns to normalize for users
        """
        self.text_model = text_model
        self.genre_vocab = genre_vocab
        self.numeric_cols = numeric_cols or ['age']

    def create_profiles(self, users_df):
        """Main entry point for creating the user profile matrix."""

        # 1) Normalize numeric columns (e.g., age)
        users_df[self.numeric_cols] = users_df[self.numeric_cols].fillna(0)
        users_df = normalize_multiple_columns(users_df, self.numeric_cols)
        numeric_fields = users_df[self.numeric_cols].to_numpy()  # shape: [num_users, len(numeric_cols)]        

        # 2) Multi-hot favorite genres
        users_df['favorite_genres'] = users_df['favorite_genres'].fillna('')
        user_genres_encoded = multi_hot_encode_genres(
            users_df,
            self.genre_vocab,
            split=',',
            genres_column='favorite_genres'
        )

        # 3) One-hot encode sex and spending category
        users_df['sex'] = users_df['sex'].fillna('Unknown')
        user_sex_one_hot = one_hot_encode_sex(users_df, 'sex')
        spending_one_hot = one_hot_encode_column(users_df, 'spending_category')

        # 4) Encode user tags
        users_df['tags_user'] = users_df['tags_user'].fillna("")
        user_tags_list = users_df['tags_user'].tolist()
        user_tag_embeddings = batch_encode_text(
            user_tags_list, self.text_model, convert_to_tensor=True
        )

        # 5) Combine main profile
        user_profiles = np.hstack([
            numeric_fields,       # shape: [num_users, 3]
            user_genres_encoded,  # shape: [num_users, len(genre_vocab)]
            spending_one_hot,     # shape: [num_users, spending_dim]
            user_sex_one_hot,
            user_tag_embeddings   # shape: [num_users, tag_dim]
        ])

        return user_profiles

class LazyJaccardMovieLensDataset(Dataset):
    """
    A memory-efficient dataset that computes the Jaccard Similarity between user and movie genres as the label.
    """
    def __init__(self, ratings, user_profiles, movie_profiles, genre_vocab):
        """
        Args:
            ratings (pd.DataFrame): DataFrame with 'userIndex', 'movieIndex', and 'rating'.
            user_profiles (np.ndarray): Precomputed user profiles (shape: [num_users, profile_dim]).
            movie_profiles (np.ndarray): Precomputed movie profiles (shape: [num_movies, profile_dim]).
            genre_vocab (list): List of genre vocabulary to extract genre labels.
        """
        self.ratings = ratings
        self.user_indices = ratings['userIndex'].values
        self.movie_indices = ratings['movieIndex'].values
        self.ratings = ratings['rating'].values

        self.user_profiles = user_profiles
        self.movie_profiles = movie_profiles
        self.genre_vocab = genre_vocab  # Used for lazy extraction of genres
        self.num_genres = len(genre_vocab)  # Number of genres

    def __len__(self):
        return len(self.ratings)

    def _extract_genres(self, profile, is_user):
        """
        Extract genres from the given profile.
        Args:
            profile (np.ndarray): User or movie profile.
            is_user (bool): Whether the profile is a user profile (True) or a movie profile (False).
        Returns:
            set: Set of genres associated with the profile.
        """
        if is_user:
            # User genres are from index 1 to 1 + num_genres
            genre_part = profile[1:1 + self.num_genres]
        else:
            # Movie genres are from index 5 + 768 to 5 + 768 + num_genres
            genre_part = profile[5 + 768:5 + 768 + self.num_genres]

        return {self.genre_vocab[i] for i, val in enumerate(genre_part) if val > 0.5}  # Threshold to determine active genres

    def _compute_jaccard(self, user_genres, movie_genres):
        """
        Compute Jaccard Similarity between user genres and movie genres.
        Args:
            user_genres (set): Set of genres for the user.
            movie_genres (set): Set of genres for the movie.
        Returns:
            float: Jaccard Similarity score.
        """
        if not user_genres or not movie_genres:
            return 0.0
        intersection = len(user_genres & movie_genres)
        union = len(user_genres | movie_genres)
        return intersection / union

    def __getitem__(self, idx):
        """
        Returns a single data point: user profile, movie profile, rating, and Jaccard Similarity label.
        """
        # Extract user and movie indices
        user_idx = self.user_indices[idx]
        movie_idx = self.movie_indices[idx]

        # Extract user and movie profiles
        user_profile = torch.tensor(self.user_profiles[user_idx], dtype=torch.float32)
        movie_profile = torch.tensor(self.movie_profiles[movie_idx], dtype=torch.float32)

        # Extract user and movie genres
        user_genres = self._extract_genres(self.user_profiles[user_idx], is_user=True)
        movie_genres = self._extract_genres(self.movie_profiles[movie_idx], is_user=False)

        # Compute Jaccard Similarity
        jaccard_label = self._compute_jaccard(user_genres, movie_genres)

        # Extract rating
        rating = torch.tensor(self.ratings[idx], dtype=torch.float32)

        return user_profile, movie_profile, rating, torch.tensor(jaccard_label, dtype=torch.float32)

class JaccardRelevanceMovieLensDataset(Dataset):
    def __init__(self, ratings, user_profiles, movie_profiles, genre_vocab, user_rated_movies):
        """
        Dataset for handling user-movie interactions with Jaccard labels and precision targets.

        Args:
            ratings (pd.DataFrame): DataFrame with 'userIndex', 'movieIndex', and 'rating'.
            user_profiles (np.ndarray): Precomputed user profiles (shape: [num_users, profile_dim]).
            movie_profiles (np.ndarray): Precomputed movie profiles (shape: [num_movies, profile_dim]).
            genre_vocab (list): List of genre vocabulary to extract genre labels.
            threshold (float): Rating threshold to determine relevance for precision_targets.
        """
        self.ratings = ratings
        self.user_indices = ratings['userIndex'].values
        self.movie_indices = ratings['movieIndex'].values
        self.ratings_values = ratings['rating'].values

        self.user_profiles = user_profiles
        self.movie_profiles = movie_profiles
        self.genre_vocab = genre_vocab
        self.user_rated_movies = user_rated_movies

        # Dynamically calculate max_rated_movies
        self.max_rated_movies = max(len(movies) for movies in user_rated_movies)

    def _extract_genres(self, profile, is_user):
        """
        Extract genres from the given profile.
        """
        if is_user:
            genre_part = profile[1:1 + len(self.genre_vocab)]
        else:
            genre_part = profile[5 + 768:5 + 768 + len(self.genre_vocab)]

        return {self.genre_vocab[i] for i, val in enumerate(genre_part) if val > 0.5}

    def _compute_jaccard(self, user_genres, movie_genres):
        """
        Compute Jaccard Similarity between user genres and movie genres.
        """
        if not user_genres or not movie_genres:
            return 0.0
        intersection = len(user_genres & movie_genres)
        union = len(user_genres | movie_genres)
        return intersection / union

    def __getitem__(self, idx):
        """
        Returns a single data point: user profile, movie profile, rating, Jaccard label, and sparse relevance tensor.
        """
        user_idx = self.user_indices[idx]
        movie_idx = self.movie_indices[idx]

        user_profile = torch.tensor(self.user_profiles[user_idx], dtype=torch.float32)
        movie_profile = torch.tensor(self.movie_profiles[movie_idx], dtype=torch.float32)
        rating = torch.tensor(self.ratings_values[idx], dtype=torch.float32)

        # Compute Jaccard similarity
        user_genres = self._extract_genres(self.user_profiles[user_idx], is_user=True)
        movie_genres = self._extract_genres(self.movie_profiles[movie_idx], is_user=False)
        genre_overlap = torch.tensor(self._compute_jaccard(user_genres, movie_genres), dtype=torch.float32)

        # Retrieve rated movies for the user
        rated_movies = self.user_rated_movies[user_idx]
        rated_movies_padded = rated_movies + [len(self.movie_profiles)] * (self.max_rated_movies - len(rated_movies))
        rated_movies_tensor = torch.tensor(rated_movies_padded, dtype=torch.long)

        return user_profile, movie_profile, rating, genre_overlap, rated_movies_tensor

    def __len__(self):
        return len(self.ratings)

class CustomMovieLensDataset(Dataset):
    def __init__(
        self, 
        ratings, 
        user_profiles, 
        movie_profiles, 
        genre_vocab,
        user_token_dims,    # e.g. {"age":1, "num_reviews":1, ...}
        movie_token_dims    # e.g. {"budget":1, "popularity":1, ...}
    ):
        """
        Dataset for handling user-movie interactions with Jaccard labels.

        Args:
            ratings (pd.DataFrame): DataFrame with columns ['userIndex', 'movieIndex', 'rating'].
            user_profiles (np.ndarray): [num_users, user_profile_dim].
            movie_profiles (np.ndarray): [num_movies, movie_profile_dim].
            genre_vocab (list): List of genre vocabulary strings.
            user_rated_movies (list of lists): Each element is a list of movies rated by a user.
            user_token_dims (dict): e.g. {"age": 1, "num_reviews": 1, ...}
            movie_token_dims (dict): e.g. {"budget": 1, "popularity": 1, ...}
        """
        self.ratings = ratings
        self.user_indices = ratings['userIndex'].values
        self.movie_indices = ratings['movieIndex'].values
        self.ratings_values = ratings['rating'].values

        self.user_profiles = user_profiles
        self.movie_profiles = movie_profiles
        self.genre_vocab = genre_vocab

        self.user_token_dims = user_token_dims
        self.movie_token_dims = movie_token_dims

    def _extract_genres(self, profile, is_user):
        """
        Extract genres from the given profile.
        """
        if is_user:
            # example: user genre bits at indices [1 : 1 + len(genre_vocab)]
            genre_part = profile[3 : 3 + len(self.genre_vocab)]
        else:
            # example: movie genre bits at indices [5+768 : 5+768+len(genre_vocab)]
            # *This is just an example offset from your snippet*
            genre_part = profile[5 + 384 : 5 + 384 + len(self.genre_vocab)]

        return {
            self.genre_vocab[i]
            for i, val in enumerate(genre_part)
            if val > 0.5
        }

    def _compute_jaccard(self, user_genres, movie_genres):
        """
        Compute Jaccard Similarity between user genres and movie genres.
        """
        if not user_genres or not movie_genres:
            return 0.0
        intersection = len(user_genres & movie_genres)
        union = len(user_genres | movie_genres)
        return intersection / union

    def _slice_tokens(self, full_profile, token_dims):
        """
        Slice a 1D profile vector into multiple tokens,
        returning a dict {token_name: slice}.
        """
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
            user_tokens, movie_tokens, rating, genre_overlap, rated_movies_tensor
        """
        user_idx = self.user_indices[idx]
        movie_idx = self.movie_indices[idx]

        # 1. Basic Tensors
        user_profile_np = self.user_profiles[user_idx]  # shape [user_profile_dim]
        movie_profile_np = self.movie_profiles[movie_idx]  # shape [movie_profile_dim]
        rating = torch.tensor(self.ratings_values[idx], dtype=torch.float32)

        # 2. Token Slicing
        user_profile_tensor = torch.tensor(user_profile_np, dtype=torch.float32)
        user_tokens = self._slice_tokens(user_profile_tensor, self.user_token_dims)
        
        movie_profile_tensor = torch.tensor(movie_profile_np, dtype=torch.float32)
        movie_tokens = self._slice_tokens(movie_profile_tensor, self.movie_token_dims)

        # 3. Jaccard Overlap
        user_genres = self._extract_genres(user_profile_np, is_user=True)
        movie_genres = self._extract_genres(movie_profile_np, is_user=False)
        genre_overlap = torch.tensor(self._compute_jaccard(user_genres, movie_genres), dtype=torch.float32)

        return user_tokens, movie_tokens, rating, genre_overlap

    def __len__(self):
        return len(self.ratings)

# Wrapping genre label generator into a Dataset subclass
class LazyGenreMovieLensDataset(Dataset):
    """
    A memory-efficient extension of ContentFeatureMovieLensDataset with lazy genre label extraction.
    """
    def __init__(self, ratings, user_profiles, movie_profiles, genre_vocab):
        """
        Args:
            ratings (pd.DataFrame): DataFrame with 'userIndex', 'movieIndex', and 'rating'.
            user_profiles (np.ndarray): Precomputed user profiles (shape: [num_users, profile_dim]).
            movie_profiles (np.ndarray): Precomputed movie profiles (shape: [num_movies, profile_dim]).
            genre_vocab (list): List of genre vocabulary to extract genre labels.
        """
        self.ratings = ratings
        self.user_indices = ratings['userIndex'].values
        self.movie_indices = ratings['movieIndex'].values
        self.ratings = ratings['rating'].values

        self.user_profiles = user_profiles
        self.movie_profiles = movie_profiles
        self.genre_vocab = genre_vocab  # Used for lazy extraction of genres

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        """
        Returns a single data point: user profile, movie profile, rating, and genre labels.
        """
        # Extract user and movie profiles
        user_idx = self.user_indices[idx]
        movie_idx = self.movie_indices[idx]

        user_profile = torch.tensor(self.user_profiles[user_idx], dtype=torch.float32)
        movie_profile = torch.tensor(self.movie_profiles[movie_idx], dtype=torch.float32)

        # Extract rating
        rating = torch.tensor(self.ratings[idx], dtype=torch.float32)

        # Lazy extraction of genre labels
        genre_labels = self.movie_profiles[movie_idx, (-len(self.genre_vocab) - 768):-768]  # Extract genre part

        return user_profile, movie_profile, rating, torch.tensor(genre_labels, dtype=torch.float32)
    
# Extract multi-hot genre labels for each movie in the same order as `ratings`
def extract_genre_labels(ratings_df, movie_profiles, genre_vocab):
    """
    Extracts genre labels for each movie based on the ratings DataFrame.
    
    Args:
        ratings_df (pd.DataFrame): Ratings DataFrame with 'movieIndex' column.
        movie_profiles (np.ndarray): Precomputed movie profiles with multi-hot genres.

    Returns:
        np.ndarray: Genre labels (multi-hot encoded) aligned with ratings DataFrame.
    """
    # Extract the indices of the movies in the ratings DataFrame
    movie_indices = ratings_df['movieIndex'].values
    
    # Select the multi-hot genre labels from `movie_profiles`
    genre_labels = movie_profiles[movie_indices][:, -len(genre_vocab):]  # Assuming genres are the last features
    
    return genre_labels

def batch_encode_text(texts, text_model, convert_to_tensor=True):
    """
    Encodes a list of strings using a text_model like SentenceTransformer in one or a few batches.
    
    Args:
        texts (list): List of strings to encode.
        text_model: SentenceTransformer (or similar) model for text embedding.
        convert_to_tensor (bool): Whether to convert embeddings to a torch.Tensor.
    
    Returns:
        np.ndarray: Encoded embeddings (shape [len(texts), embedding_dim]).
    """
    # Encode all texts in one shot (or you can split into mini-batches if memory is a concern)
    embeddings_torch = text_model.encode(texts, convert_to_tensor=convert_to_tensor)
    
    # Move to CPU and convert to NumPy
    embeddings_np = embeddings_torch.cpu().numpy()
    return embeddings_np

def normalize_multiple_columns(df, col_list):
    """
    Vectorized normalization of multiple numeric columns at once using MinMaxScaler.
    This performs a single fit over all columns, or each column individually in a vectorized manner.
    """
    scaler = MinMaxScaler()
    df[col_list] = scaler.fit_transform(df[col_list])
    return df

def multi_hot_encode_genres(df, genre_vocab, split='|', genres_column='genres'):
    """
    Multi-hot encode genres.
    Args:
        movies_df (pd.DataFrame): DataFrame with a 'genres' column (pipe-separated).
        all_genres (list): List of all unique genres.
    Returns:
        np.ndarray: Multi-hot encoded genres.
    """
    genre_to_index = {genre: i for i, genre in enumerate(genre_vocab)}
    multi_hot = np.zeros((len(df), len(genre_vocab)), dtype=int)

    for i, row in df.iterrows():
        movie_genres = [genre.strip() for genre in row[genres_column].split(split)]
        for genre in movie_genres:
            if genre in genre_to_index:
                multi_hot[i, genre_to_index[genre]] = 1

    return multi_hot

def one_hot_encode_sex(dataframe, column_name):
    """One-hot encode the 'sex' column (e.g., male, female, unknown)."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    return encoder.fit_transform(dataframe[[column_name]])

def scale_one_hot_to_max(one_hot_matrix, profiles):
    """
    Scale one-hot encoded features to match the maximum absolute value of the other features.
    Args:
        one_hot_matrix (np.ndarray): One-hot encoded features.
        profiles (np.ndarray): User or movie profiles (to determine max value of other features).
    Returns:
        np.ndarray: Scaled one-hot encoded features.
    """
    
    # Compute the maximum absolute value in the profiles
    max_abs_value = np.max(np.abs(profiles))

    # Scale the one-hot encoded matrix
    scaled_one_hot = one_hot_matrix * max_abs_value

    return scaled_one_hot

def normalize_one_hot_features(one_hot_matrix, scale_factor=0.2):
    """Normalize one-hot encoded features to match the scale of other features."""
    return one_hot_matrix * scale_factor

def one_hot_encode_column(dataframe, column_name):
    """One-hot encode a categorical column."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoded = encoder.fit_transform(dataframe[[column_name]])
    return one_hot_encoded

def encode_text(texts, model, batch_size=32, device='cuda'):
    """Generate embeddings for a list of texts."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, batch_size=batch_size, device=device, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def normalize_numeric_column(dataframe, column_name):
    """Normalize a numeric column using Min-Max scaling."""
    scaler = MinMaxScaler()
    dataframe[column_name] = scaler.fit_transform(dataframe[[column_name]])
    return dataframe

def scale_multi_hot_to_embedding(multi_hot_matrix, embedding_matrix):
    """
    Scale multi-hot encoded features to match the absolute maximum value of an embedding matrix.
    Args:
        multi_hot_matrix (np.ndarray): Multi-hot encoded features (e.g., genres).
        embedding_matrix (np.ndarray): Embedding matrix (e.g., overview embeddings).
    Returns:
        np.ndarray: Scaled multi-hot encoded features.
    """
    # Compute the maximum absolute value in the embedding matrix
    max_abs_value = np.max(np.abs(embedding_matrix))

    # Scale the multi-hot encoded matrix
    scaled_multi_hot = multi_hot_matrix * max_abs_value

    return scaled_multi_hot