import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-mpnet-base-v2" # "distilbert-base-uncased"

# Function to inspect the structure of a dictionary
def inspect_dictionary(dictionary, num_items=5):
    # Print the type of the dictionary
    print(f"Type: {type(dictionary)}")

    # Get the first `num_items` keys and their values
    keys = list(dictionary.keys())[:num_items]
    print(f"\nFirst {num_items} keys:")
    print(keys)

    print(f"\nFirst {num_items} values (shapes and examples):")
    for key in keys:
        value = dictionary[key]
        # Print the key and a summary of the value
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(f"Key: {key}, Value Shape: {value.shape}, Value Sample: {value[:5]}")
        elif isinstance(value, list):
            print(f"Key: {key}, Value Length: {len(value)}, Value Sample: {value[:5]}")
        else:
            print(f"Key: {key}, Value: {value}")

def split_data_by_user(ratings: pd.DataFrame):
    """
    Split data into train, validation, and test sets on a per-user basis.
    """
    train_splits = []
    val_splits = []
    test_splits = []

    for user_id, user_df in ratings.groupby('user_id'):
        user_df = user_df.sort_values('timestamp')
        num_ratings = len(user_df)
        if num_ratings < 3:
            train_splits.append(user_df)
            continue
        
        train_end = int(0.7 * num_ratings)
        val_end = int(0.85 * num_ratings)

        user_train = user_df.iloc[:train_end]
        user_val = user_df.iloc[train_end:val_end]
        user_test = user_df.iloc[val_end:]

        train_splits.append(user_train)
        val_splits.append(user_val)
        test_splits.append(user_test)

    train_data = pd.concat(train_splits)
    val_data = pd.concat(val_splits)
    test_data = pd.concat(test_splits)
    return train_data, val_data, test_data

class DataManager:
    """
    Responsible for loading and encoding user-item interaction data.
    """
    def __init__(self, ratings_path: str):
        self.ratings_path = ratings_path
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def load_ratings(self) -> pd.DataFrame:
        ratings = pd.read_csv(self.ratings_path)
        return ratings

    def encode_interactions(self, ratings: pd.DataFrame) -> pd.DataFrame:
        ratings['user_id'] = self.user_encoder.fit_transform(ratings['userId'])
        ratings['item_id'] = self.item_encoder.fit_transform(ratings['movieId'])
        return ratings

    def get_user_item_count(self, ratings: pd.DataFrame):
        n_users = ratings['user_id'].nunique()
        n_items = ratings['item_id'].nunique()
        return n_users, n_items

class UserFeatureEngineeringUsingSBERT:
    """
    Responsible for:
    - Loading enriched user data (user_enriched.csv)
    - Ensuring required fields
    - Combining fields into textual features for each user
    - Extracting embeddings from a pretrained sentence transformer
    """
    def __init__(self, user_enriched_path: str, user_encoder: LabelEncoder, device: torch.device):
        self.user_enriched_path = user_enriched_path
        self.user_encoder = user_encoder
        self.device = device

        # Fields we will use to create a natural language description of the user
        self.fields_to_use = ['age', 'sex', 'favorite_genres', 'num_reviews', 'avg_rating', 'spending_category']
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.model.to(self.device)

    def load_enriched_users(self, valid_user_ids: np.ndarray) -> pd.DataFrame:
        users_enriched = pd.read_csv(self.user_enriched_path)
        return users_enriched[users_enriched['userId'].isin(valid_user_ids)]

    def prepare_combined_features(self, users_enriched: pd.DataFrame) -> pd.DataFrame:
        # Ensure fields exist and are non-null
        for col in self.fields_to_use:
            if col not in users_enriched.columns:
                users_enriched[col] = ''
            users_enriched[col] = users_enriched[col].fillna('')

        def format_user_features(row):
            # Create a natural language representation of the user's attributes
            # Example: "User is a male, 35 years old, loves Comedy and Action. 
            # Has written 50 reviews, average rating 3.5, spending category: Medium."
            age_str = f"{int(row['age'])} years old" if str(row['age']).strip() else ""
            sex_str = f"{row['sex']}" if row['sex'].strip() else ""
            fav_genres_str = f"favorite genres: {row['favorite_genres']}" if row['favorite_genres'].strip() else ""
            num_reviews_str = f"has written {row['num_reviews']} reviews" if str(row['num_reviews']).strip() else ""
            avg_rating_str = f"with an average rating of {row['avg_rating']}" if str(row['avg_rating']).strip() else ""
            spending_str = f"spending category: {row['spending_category']}" if row['spending_category'].strip() else ""

            # Combine parts into sentences
            parts = []
            # Basic user description
            user_intro = "User is"
            if sex_str:
                user_intro += f" {sex_str},"
            if age_str:
                user_intro += f" {age_str},"
            parts.append(user_intro.strip(',') + '.')

            # Favorite genres
            if fav_genres_str:
                parts.append(f"The user has {fav_genres_str}.")

            # Activity and ratings
            if num_reviews_str or avg_rating_str:
                activity_str = " ".join([p for p in [num_reviews_str, avg_rating_str] if p]).strip()
                parts.append(f"The user {activity_str}.")

            # Spending category
            if spending_str:
                parts.append(f"The user's {spending_str}.")

            combined = " ".join(parts)
            return combined.strip()

        users_enriched['combined_features'] = users_enriched.apply(format_user_features, axis=1)
        return users_enriched

    def get_embeddings_for_texts(self, texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts, batch_size=batch_size, device=self.device, show_progress_bar=True)
            embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings)
        return embeddings

    def create_user_features(self, users_enriched: pd.DataFrame) -> dict:
        all_texts = users_enriched['combined_features'].tolist()
        user_content_features = self.get_embeddings_for_texts(all_texts)
        user_id_to_content = {
            self.user_encoder.transform([userId])[0]: content
            for userId, content in zip(users_enriched['userId'], user_content_features)
        }
        return user_id_to_content, user_content_features

class ContentFeatureEngineeringUsingSBERT:
    """
    Responsible for:
    - Loading enriched movie data
    - Ensuring required fields
    - Combining fields into textual features
    - Extracting embeddings from a pretrained language model
    """
    def __init__(self, movies_enriched_path: str, item_encoder: LabelEncoder, device: torch.device):
        self.movies_enriched_path = movies_enriched_path
        self.item_encoder = item_encoder
        self.device = device
        self.fields_to_use = [
            'genres', 'title', 'tag', 'origin_country',
            'original_language', 'original_title', 'overview', 'release_date', 'tagline'
        ]
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.model.to(self.device)

    def load_enriched_movies(self, valid_movie_ids: np.ndarray) -> pd.DataFrame:
        movies_enriched = pd.read_csv(self.movies_enriched_path)
        return movies_enriched[movies_enriched['movieId'].isin(valid_movie_ids)]

    def prepare_combined_features(self, movies_enriched: pd.DataFrame) -> pd.DataFrame:
        # Ensure fields exist and are non-null
        for col in self.fields_to_use:
            if col not in movies_enriched.columns:
                movies_enriched[col] = ''
            movies_enriched[col] = movies_enriched[col].fillna('')

        # A helper function to format the combined features for each row
        def format_combined_features(row):
            genres = row['genres'].replace('|', ' and ')
            genres_str = f"Genres: {genres}" if genres.strip() else ""
            tag_str = f"Tag: {row['tag']}" if row['tag'].strip() else ""
            title_str = f"Title: {row['title']}" if row['title'].strip() else ""
            original_title_str = (
                f"Original Title: {row['original_title']}" if row['original_title'].strip() and row['original_title'] != row['title'] else ""
            )
            country_str = f"Country: {row['origin_country']}" if row['origin_country'].strip() else ""
            language_str = f"Language: {row['original_language']}" if row['original_language'].strip() else ""
            release_date_str = ""
            if row['release_date'].strip():
                try:
                    parsed_date = pd.to_datetime(row['release_date'], errors='coerce')
                    if pd.notnull(parsed_date):
                        release_date_str = f"Release Date: {parsed_date.strftime('%B %d, %Y')}"
                    else:
                        release_date_str = f"Release Date: {row['release_date']}"
                except:
                    release_date_str = f"Release Date: {row['release_date']}"
            overview_str = f"Overview: {row['overview']}" if row['overview'].strip() else ""
            tagline_str = f"Tagline: {row['tagline']}" if row['tagline'].strip() else ""
            parts = [title_str, genres_str, country_str, language_str, original_title_str, overview_str, release_date_str, tagline_str, tag_str]
            combined = ". ".join([p for p in parts if p.strip()])
            return combined.strip()

        movies_enriched['combined_features'] = movies_enriched.apply(format_combined_features, axis=1)
        return movies_enriched

    def get_embeddings_for_texts(self, texts, batch_size=32):
        """
        Get sentence embeddings using SentenceTransformer
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts, batch_size=batch_size, device=self.device, show_progress_bar=True)
            embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings)
        return embeddings

    def create_content_features(self, movies_enriched: pd.DataFrame) -> dict:
        all_texts = movies_enriched['combined_features'].tolist()
        item_content_features = self.get_embeddings_for_texts(all_texts)
        movie_id_to_content = {
            self.item_encoder.transform([movieId])[0]: content
            for movieId, content in zip(movies_enriched['movieId'], item_content_features)
        }
        return movie_id_to_content, item_content_features


class FeatureEngineeringUsingBERT:
    """
    Responsible for:
    - Loading enriched movie data
    - Ensuring required fields
    - Combining fields into textual features
    - Extracting embeddings from a pretrained language model
    """
    def __init__(self, movies_enriched_path: str, item_encoder: LabelEncoder, device: torch.device):
        self.movies_enriched_path = movies_enriched_path
        self.item_encoder = item_encoder
        self.device = device
        self.fields_to_use = [
            'genres', 'title', 'tag', 'origin_country',
            'original_language', 'original_title', 'overview', 'release_date', 'tagline'
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.transformer_model = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)

    def load_enriched_movies(self, valid_movie_ids: np.ndarray) -> pd.DataFrame:
        movies_enriched = pd.read_csv(self.movies_enriched_path)
        return movies_enriched[movies_enriched['movieId'].isin(valid_movie_ids)]

    def prepare_combined_features(self, movies_enriched: pd.DataFrame) -> pd.DataFrame:
        # Ensure fields exist and are non-null
        for col in self.fields_to_use:
            if col not in movies_enriched.columns:
                movies_enriched[col] = ''
            movies_enriched[col] = movies_enriched[col].fillna('')

        # A helper function to format the combined features for each row
        def format_combined_features(row):
            # Replace '|' with ' and ' in genres
            genres = row['genres'].replace('|', ' and ')
            genres_str = f"Genres: {genres}" if genres.strip() else ""

            # Tag: If you want to keep tags, label them clearly (optional)
            tag_str = f"Tag: {row['tag']}" if row['tag'].strip() else ""

            # Title: Always show main title
            title_str = f"Title: {row['title']}" if row['title'].strip() else ""

            # If original_title differs from title, include it
            original_title_str = ""
            if row['original_title'].strip() and row['original_title'] != row['title']:
                original_title_str = f"Original Title: {row['original_title']}"

            # Country and Language
            country_str = f"Country: {row['origin_country']}" if row['origin_country'].strip() else ""
            language_str = f"Language: {row['original_language']}" if row['original_language'].strip() else ""

            # Normalize release_date if present
            release_date_str = ""
            if row['release_date'].strip():
                # Attempt to parse the date
                try:
                    parsed_date = pd.to_datetime(row['release_date'], errors='coerce')
                    if pd.notnull(parsed_date):
                        release_date_str = f"Release Date: {parsed_date.strftime('%B %d, %Y')}"
                    else:
                        # If parsing fails, fall back to the raw value
                        release_date_str = f"Release Date: {row['release_date']}"
                except:
                    release_date_str = f"Release Date: {row['release_date']}"

            # Overview
            overview_str = f"Overview: {row['overview']}" if row['overview'].strip() else ""

            # Tagline
            tagline_str = f"Tagline: {row['tagline']}" if row['tagline'].strip() else ""

            # Combine all parts into a natural, coherent text
            # Filter out empty strings
            parts = [title_str, genres_str, country_str, language_str, original_title_str, overview_str, release_date_str, tagline_str, tag_str]
            combined = ". ".join([p for p in parts if p.strip()])

            return combined.strip()

        movies_enriched['combined_features'] = movies_enriched.apply(format_combined_features, axis=1)
        return movies_enriched


    def get_embeddings_for_texts(self, texts, batch_size=32):
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self.transformer_model(**encoded)
                hidden_states = output.last_hidden_state
                batch_embeddings = hidden_states.mean(dim=1)  # mean pooling
                embeddings.append(batch_embeddings.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    def create_content_features(self, movies_enriched: pd.DataFrame) -> dict:
        all_texts = movies_enriched['combined_features'].tolist()
        item_content_features = self.get_embeddings_for_texts(all_texts)
        movie_id_to_content = {
            self.item_encoder.transform([movieId])[0]: content
            for movieId, content in zip(movies_enriched['movieId'], item_content_features)
        }
        return movie_id_to_content, item_content_features
    
class MovieLensDataset(Dataset):
    """
    PyTorch Dataset for MovieLens, now incorporating user content embeddings.
    """
    def __init__(self, ratings, user_content_features_dict, item_content_features_dict):
        self.users = torch.tensor(ratings['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(ratings['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings['rating'].values, dtype=torch.float32)
        
        # Build tensors for user and item content features
        self.user_content_features = torch.tensor(
            np.array([user_content_features_dict[u] for u in ratings['user_id']]),
            dtype=torch.float32
        )
        
        self.item_content_features = torch.tensor(
            np.array([item_content_features_dict[i] for i in ratings['item_id']]),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        # Return user_id, item_id, user_content_embed, item_content_embed, rating
        return (self.users[idx], 
                self.items[idx], 
                self.user_content_features[idx], 
                self.item_content_features[idx], 
                self.ratings[idx])

