import torch
import numpy as np
import os
from codebase.v6.models import TransformerRULModel

def get_model(sequence_length, tabular_dataset, save_path):
    """
    Create or load a new model instance for a specific sequence length.
    You can adapt this function to your own model initialization needs.
    """
    assert sequence_length in [30, 60, 90, 120], f"Invalid sequence length: {sequence_length}"

    input_dim = len(tabular_dataset.feature_cols)  # Number of input features

    # Load the model config from disk (must match training config)
    model_config = torch.load(
        os.path.join(save_path, f"model_config_seq_len_{sequence_length}.pth"),
        map_location="cpu", weights_only=True
    )

    model = TransformerRULModel(
        input_dim=input_dim,
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=model_config["dropout"]
    )

    return model, model_config


def predict_for_sequence(model, val_loader, device="cpu", sequence_position="last"):
    """
    Evaluate the model on the validation set, returning predictions for either:
      - 'first' sequence of each engine, or
      - 'last' sequence of each engine,
    depending on `sequence_position`.

    Returns a dict: {engine_id: predicted_rul}.
    """
    model.eval()
    predictions = {}
    visited_engines = set()

    with torch.no_grad():
        for (features, _, masks, engine_ids, cycles) in val_loader:
            # Move tensors to device
            features, masks = features.to(device), masks.to(device)
            outputs = model(features, masks)  # (batch_size,)

            for i, engine_id in enumerate(engine_ids):
                e_id = int(engine_id.item())

                # If we want the "last" sequence predictions:
                if sequence_position == "last":                    
                    if e_id not in visited_engines:
                        visited_engines.add(e_id)
                        predictions[e_id] = outputs[i].item()

                    else:                        
                        predictions[e_id] = outputs[i].item() # overwrite last prediction                        

                elif sequence_position == "first":
                    # If we want the first sequence for each engine
                    # We'll define "first" by checking if the min_cycle_in_seq is the earliest known for this engine
                    if e_id not in visited_engines:
                        visited_engines.add(e_id)
                        predictions[e_id] = outputs[i].item()

                    continue
                    
                else:
                    raise ValueError(f"Unknown sequence_position: {sequence_position}")

    return predictions


def get_val_rul_for_each_sequence_position(loaders_by_sequence_length):
    """
    Extracts the true RUL for both the first and last sequence of each engine in the validation set.

    Returns two dicts:
      - first_seq_rul:  {engine_id: true_rul_of_first_seq}
      - last_seq_rul:   {engine_id: true_rul_of_last_seq}
    """
    first_seq_rul = {}
    last_seq_rul = {}

    for _, fold_loaders in loaders_by_sequence_length.items():
        (_, val_loader) = fold_loaders[0]  # Use fold 0 as an example

        for _, targets, _, engine_ids, _ in val_loader:
            # For each item in batch, decide if it's the first or last sequence
            for i in range(len(engine_ids)):
                e_id = int(engine_ids[i].item())

                rul_value = targets[i].item()

                # If we haven't stored a first_seq_rul for this engine or found an earlier sequence:
                if e_id not in first_seq_rul:
                    first_seq_rul[e_id] = rul_value                
                
                if e_id not in last_seq_rul:
                    last_seq_rul[e_id] = 0                

    return first_seq_rul, last_seq_rul


def gather_base_model_predictions_first_and_last(
    tabular_dataset,
    loaders_by_sequence_length,
    sequence_lengths,
    save_path,
    device="cpu"
):
    """
    Gathers predictions from each base model for BOTH the first and last sequences of each engine,
    building a single row [model1_first, model2_first, ..., model4_first, model1_last, ..., model4_last].

    Returns:
        X_val (np.ndarray): shape (num_engines_val, 2 * len(sequence_lengths)) for first & last predictions
        y_val_first (np.ndarray): shape (num_engines_val,) ground truth RUL for the first sequence
        y_val_last (np.ndarray): shape (num_engines_val,) ground truth RUL for the last sequence (often near 0)
        engine_ids (list): list of engine IDs
    """
    # 1. Load base models and evaluate first+last predictions
    predictions_first_by_seq_len = {}
    predictions_last_by_seq_len = {}

    # Evaluate for each seq_len
    for seq_len in sequence_lengths:
        (_, val_loader) = loaders_by_sequence_length[seq_len][0]  # fold 0

        # Load model
        model, _ = get_model(seq_len, tabular_dataset=tabular_dataset, save_path=save_path)
        
        model_ckpt = torch.load(
            f"{save_path}/model_seq_len_{seq_len}.pth",
            map_location=device
        )
        model.load_state_dict(model_ckpt["model_state_dict"])
        model.to(device)

        print(f"Gathering FIRST sequence predictions, seq_len={seq_len}")
        preds_first = predict_for_sequence(
            model, val_loader, device=device, sequence_position="first"
        )

        print(f"Gathering LAST sequence predictions, seq_len={seq_len}")
        preds_last = predict_for_sequence(
            model, val_loader, device=device, sequence_position="last"
        )

        predictions_first_by_seq_len[seq_len] = preds_first
        predictions_last_by_seq_len[seq_len]  = preds_last

    # 2. Gather ground-truth for first & last sequences
    first_rul_dict, last_rul_dict = get_val_rul_for_each_sequence_position(loaders_by_sequence_length)

    # 3. Build a consistent list of engine IDs
    all_engine_ids = set()
    for seq_len in sequence_lengths:
        all_engine_ids.update(predictions_first_by_seq_len[seq_len].keys())
        all_engine_ids.update(predictions_last_by_seq_len[seq_len].keys())
        
    all_engine_ids.update(first_rul_dict.keys())
    all_engine_ids.update(last_rul_dict.keys())

    engine_ids = sorted(all_engine_ids)

    # 4. Construct the feature matrix (X_val) and target arrays (y_val_first, y_val_last)
    X_val = []
    y_val_first = []
    y_val_last  = []

    for e_id in engine_ids:
        row = []
        # For each seq_len, gather first-sequence pred
        for seq_len in sequence_lengths:
            row.append(predictions_first_by_seq_len[seq_len].get(e_id, np.nan))

        # For each seq_len, gather last-sequence pred
        for seq_len in sequence_lengths:
            row.append(predictions_last_by_seq_len[seq_len].get(e_id, np.nan))

        X_val.append(row)

        # Gather ground truth RULs
        # If an engine doesn't have an entry, fallback to np.nan
        first_rul_value = first_rul_dict.get(e_id, np.nan)
        last_rul_value  = last_rul_dict.get(e_id, np.nan)

        y_val_first.append(first_rul_value)
        y_val_last.append(last_rul_value)

    X_val = np.array(X_val, dtype=np.float32)       # shape: (num_engines, 2 * len(sequence_lengths))
    y_val_first = np.array(y_val_first, dtype=np.float32)  # (num_engines,)
    y_val_last  = np.array(y_val_last, dtype=np.float32)   # (num_engines,)

    return X_val, y_val_first, y_val_last, engine_ids