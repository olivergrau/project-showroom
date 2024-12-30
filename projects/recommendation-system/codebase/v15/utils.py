# utils.py
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    user_tokens, movie_tokens, ratings, genre_overlap = zip(*batch)

    # Stack user tokens
    user_token_stacked = {
        key: torch.stack([u[key] for u in user_tokens]) for key in user_tokens[0]
    }

    # Stack movie tokens
    movie_token_stacked = {
        key: torch.stack([m[key] for m in movie_tokens]) for key in movie_tokens[0]
    }

    # Stack dense inputs
    ratings = torch.stack(ratings)
    genre_overlap = torch.stack(genre_overlap)

    return user_token_stacked, movie_token_stacked, ratings, genre_overlap
