# ----------------------------------------------------------------
# Profile Creator Interfaces
# ----------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np

class IProfileCreator:
    """Interface for profile creators."""

    def create_profiles(self, df, *args, **kwargs):
        """Generate numeric embeddings (profiles) from a DataFrame."""
        raise NotImplementedError

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
        text_model_tags : SentenceTransformer or similar
            Model to encode textual data for tags.
        text_model_plots : SentenceTransformer or similar
            Model to encode textual data for overviews.
        genre_vocab : list
            All unique genres extracted from user/movie data.
        numeric_cols : list
            Numeric columns to normalize for movies.
        """
        self.text_model_tags = text_model_tags
        self.text_model_plots = text_model_plots
        self.genre_vocab = genre_vocab
        self.numeric_cols = numeric_cols or [
            'movieId', 'budget', 'popularity', 'runtime', 'vote_average', 'vote_count'
        ]

    def create_profiles(self, movies_df):
        """
        Main entry point for creating the movie profile matrix.

        Args:
            movies_df (pd.DataFrame): Input dataframe containing movie features.

        Returns:
            tuple: 
                - movie_profiles (np.ndarray): Normalized movie profile matrix.
                - movieId_mapping (dict): Mapping from real movie IDs to normalized movie IDs.
        """

        original_movie_ids = movies_df["movieId"].copy().to_numpy()  # Original movie IDs

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
        movies_df, _ = normalize_multiple_columns(
            movies_df, self.numeric_cols
        )

        # Extract numeric features (normalized)
        numeric_features = movies_df[self.numeric_cols].to_numpy()  # shape: [num_movies, len(numeric_cols)]

        # Create mapping for movieId normalization
        normalized_movie_ids = numeric_features[:, 0]          # Normalized movie IDs
        movieId_mapping = {
            int(original_id): float(normalized_id)
            for original_id, normalized_id in zip(original_movie_ids, normalized_movie_ids)
        }

        # 5) Combine all features
        movie_profiles = np.hstack([
            numeric_features,          # shape: [num_movies, len(numeric_cols)]
            movie_overview_embeddings, # shape: [num_movies, overview_dim]
            movie_genres_encoded,      # shape: [num_movies, len(genre_vocab)]
            movie_tag_embeddings       # shape: [num_movies, tag_dim]
        ])

        return movie_profiles, movieId_mapping

class UserProfileCreator(IProfileCreator):
    """
    Responsible for creating user profiles by combining:
      - Normalized numeric features (e.g., age)
      - Multi-hot favorite genres
      - Tag embeddings
      - one-hot sex
    """

    def __init__(self, text_model, genre_vocab, numeric_cols=None):
        """
        Parameters:
        -----------
        text_model : SentenceTransformer or similar
            Model to encode textual data.
        genre_vocab : list
            All unique genres extracted from user/movie data.
        numeric_cols : list
            Numeric columns to normalize for users.
        """
        self.text_model = text_model
        self.genre_vocab = genre_vocab
        self.numeric_cols = numeric_cols or ['userId', 'age', 'num_reviews', 'avg_rating']

    def create_profiles(self, users_df):
        """
        Main entry point for creating the user profile matrix.

        Args:
            users_df (pd.DataFrame): Input dataframe containing user features.

        Returns:
            tuple:
                - user_profiles (np.ndarray): Normalized user profile matrix.
                - userId_mapping (dict): Mapping from real user IDs to normalized user IDs.
        """

        original_user_ids = users_df["userId"].copy().to_numpy()  # Original user IDs

        # 1) Normalize numeric columns (e.g., userId, age, num_reviews)
        users_df[self.numeric_cols] = users_df[self.numeric_cols].fillna(0)
        users_df, _ = normalize_multiple_columns(
            users_df, self.numeric_cols
        )

        # Extract numeric fields (normalized)
        numeric_fields = users_df[self.numeric_cols].to_numpy()  # shape: [num_users, len(numeric_cols)]

        # Create mapping for userId normalization
        normalized_user_ids = numeric_fields[:, 0]          # Normalized user IDs
        userId_mapping = {
            int(original_id): float(normalized_id)
            for original_id, normalized_id in zip(original_user_ids, normalized_user_ids)
        }

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
        user_sex_one_hot = one_hot_encode_sex(users_df, 'sex')  # e.g., ['Male', 'Female', 'Unknown']
        spending_one_hot = one_hot_encode_column(users_df, 'spending_category')  # Spending categories

        # 4) Encode user tags
        users_df['tags_user'] = users_df['tags_user'].fillna("")
        user_tags_list = users_df['tags_user'].tolist()
        user_tag_embeddings = batch_encode_text(
            user_tags_list, self.text_model, convert_to_tensor=True
        )

        # 5) Combine all features
        user_profiles = np.hstack([
            numeric_fields,       # shape: [num_users, len(numeric_cols)]
            user_genres_encoded,  # shape: [num_users, len(genre_vocab)]
            spending_one_hot,     # shape: [num_users, spending_dim]
            user_sex_one_hot,     # shape: [num_users, num_sex_classes]
            user_tag_embeddings   # shape: [num_users, tag_dim]
        ])

        return user_profiles, userId_mapping
    
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
    return df, scaler

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