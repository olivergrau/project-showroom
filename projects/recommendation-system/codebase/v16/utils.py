import torch

def collate_fn(batch):
    """
    Custom collate function to handle batching of:
    - user_tokens (dict of tensors)
    - movie_tokens (dict of tensors)
    - ratings, genre_overlap (dense scalars)
    - user_history_tensor (padded tensor)
    - user_history_mask (mask tensor)
    """
    user_tokens, movie_tokens, ratings, genre_overlap, user_histories, user_masks = zip(*batch)

    # Stack user tokens
    user_token_stacked = {
        key: torch.stack([u[key] for u in user_tokens]) for key in user_tokens[0]
    }

    # Stack movie tokens
    movie_token_stacked = {
        key: torch.stack([m[key] for m in movie_tokens]) for key in movie_tokens[0]
    }

    # Stack dense inputs
    ratings = torch.stack(ratings)  # Shape: [batch_size]
    genre_overlap = torch.stack(genre_overlap)  # Shape: [batch_size]

    # Stack user histories
    user_history_stacked = {
        key: torch.stack([h[key] for h in user_histories]) for key in user_histories[0]
    }
    user_mask_stacked = torch.stack(user_masks)  # Shape: [batch_size, max_history_items]

    return user_token_stacked, movie_token_stacked, ratings, genre_overlap, user_history_stacked, user_mask_stacked