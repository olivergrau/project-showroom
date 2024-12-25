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

    def __init__(self, text_model, genre_vocab, numeric_cols=None):
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
        self.text_model = text_model
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
            overview_list, self.text_model, convert_to_tensor=True
        )

        # 3) Text embeddings: Tags
        movies_df['tags_by_users'] = movies_df['tags_by_users'].fillna("")
        tags_list = movies_df['tags_by_users'].tolist()
        movie_tag_embeddings = batch_encode_text(
            tags_list, self.text_model, convert_to_tensor=True
        )

        # 4) Normalize numeric columns
        movies_df[self.numeric_cols] = movies_df[self.numeric_cols].fillna(0)
        movies_df = normalize_multiple_columns(movies_df, self.numeric_cols)
        numeric_features = movies_df[self.numeric_cols].to_numpy()  # shape: [num_movies, 5]

        movie_genres_scaled = scale_multi_hot_to_embedding(movie_genres_encoded, movie_tag_embeddings)

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
      - Scaled one-hot sex
    """

    def __init__(self, text_model, genre_vocab, numeric_cols=None):
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
        age_array = users_df[self.numeric_cols].to_numpy()  # shape: [num_users, 1]

        # 2) Multi-hot favorite genres
        users_df['favorite_genres'] = users_df['favorite_genres'].fillna('')
        user_genres_encoded = multi_hot_encode_genres(
            users_df,
            self.genre_vocab,
            split=',',
            genres_column='favorite_genres'
        )

        # 3) One-hot encode sex
        users_df['sex'] = users_df['sex'].fillna('Unknown')
        user_sex_one_hot = one_hot_encode_sex(users_df, 'sex')

        # 4) Encode user tags
        users_df['tags_user'] = users_df['tags_user'].fillna("")
        user_tags_list = users_df['tags_user'].tolist()
        user_tag_embeddings = batch_encode_text(
            user_tags_list, self.text_model, convert_to_tensor=True
        )

        user_genres_scaled = scale_multi_hot_to_embedding(user_genres_encoded, user_tag_embeddings)

        # 5) Combine main profile
        user_profiles_without_one_hot = np.hstack([
            age_array,            # shape: [num_users, 1]
            user_genres_encoded,  # shape: [num_users, len(genre_vocab)]
            user_tag_embeddings   # shape: [num_users, tag_dim]
        ])

        # 6) Scale the one-hot (sex) feature
        user_sex_scaled = scale_one_hot_to_max(user_sex_one_hot, user_profiles_without_one_hot)

        # 7) Final concatenation
        user_profiles = np.hstack([
            user_profiles_without_one_hot,
            user_sex_scaled
        ])

        return user_profiles
    
class EvaluationDataset(Dataset):
    """
    Dataset that enumerates (userIndex, movieIndex) for all test users x all movies.
    """
    def __init__(self, test_data, user_profiles, movie_profiles):
        """
        Args:
            test_data (pd.DataFrame): Has userIndex, movieIndex, rating columns (row-by-row).
            user_profiles (np.ndarray): [num_users, user_dim].
            movie_profiles (np.ndarray): [num_movies, movie_dim].
        """
        self.user_profiles = user_profiles
        self.movie_profiles = movie_profiles

        # 1) Which users appear in the test set?
        self.unique_test_users = test_data['userIndex'].unique()

        # 2) All valid movies we might recommend from
        #    Typically, this is the entire set of movie embeddings you have.
        self.all_movie_indices = np.arange(len(movie_profiles))

        # 3) Build pairs: (userIndex, movieIndex)
        self.eval_pairs = []
        for user_idx in self.unique_test_users:
            for movie_idx in self.all_movie_indices:
                self.eval_pairs.append((user_idx, movie_idx))

    def __len__(self):
        return len(self.eval_pairs)

    def __getitem__(self, idx):
        user_idx, movie_idx = self.eval_pairs[idx]
        user_profile = torch.tensor(self.user_profiles[user_idx], dtype=torch.float32)
        movie_profile = torch.tensor(self.movie_profiles[movie_idx], dtype=torch.float32)
        
        # Return userIndex/movieIndex too, so we can group results
        return user_profile, movie_profile, user_idx, movie_idx

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