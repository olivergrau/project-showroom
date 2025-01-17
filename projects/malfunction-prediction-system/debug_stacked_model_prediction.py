import torch
import numpy as np
import sys
from codebase.v6.models import TransformerRULModel
import os
import pandas as pd
import torch
from codebase.v6.data import prepare_loaders_for_sequence_lengths, prepare_sequence_dataloader
from codebase.v6.dataset import CMAPSSDataset

import torch
import numpy as np
import os
from codebase.v6.models import TransformerRULModel

def get_model(sequence_length):
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
        model, _ = get_model(seq_len)
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

if __name__ == "__main__":

    # Save the model state dictionary
    save_path = "saved_weights/"
    feature_cols_path = os.path.join(save_path, "feature_cols.pth")
    model_path = os.path.join(save_path, "model.pth")
    scaler_path = os.path.join(save_path, "scaler.pth")

    # Set up paths and features
    data_dir = "data/CMAPSSData"
    data_index = 1  # Choose FD001, FD002, etc.

    compute_engineered_features = True

    train_data = pd.read_csv("./data/CMAPSSData/train_FD001.txt", sep=r'\s+', header=None)
    train_data.columns = ['engine_id', 'cycle'] + [f'os_{i+1}' for i in range(3)] + [f'sensor_{i}' for i in range(1, 22)]

    feature_cols = [f'os_{i+1}' for i in range(3)] + [f'sensor_{i}' for i in range(1, 22)]

    constant_cols = train_data[feature_cols].nunique()
    constant_cols = constant_cols[constant_cols == 1]

    feature_cols = [col for col in feature_cols if col not in constant_cols.index.tolist()]

    torch.save(feature_cols, feature_cols_path)
    print(f"Feature columns saved to {feature_cols_path}")

    scaler = None

    # load existing scaler
    if os.path.exists(scaler_path):
        scaler = torch.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")

    # Initialize the dataset
    tabular_dataset = CMAPSSDataset(
        data_dir=data_dir,
        data_index=data_index,
        feature_cols=feature_cols,
        mode="train",
        compute_engineered_features=compute_engineered_features,
        features_to_engineer=["sensor_12", "sensor_7", "sensor_21", "sensor_20", "sensor_11", "sensor_4", "sensor_15"],
        scaler=scaler
    )

    # save the scaler (all derived datasets use this scaler)
    torch.save(tabular_dataset.scaler, scaler_path)

    # Example usage
    overlap_factor = 0.5
    sequence_lengths = [30, 60, 90, 120]
    n_splits = 5
    batch_size = 256

    loaders_by_sequence_length = prepare_loaders_for_sequence_lengths(
        sequence_lengths=sequence_lengths,
        tabular_dataset=tabular_dataset,
        n_splits=n_splits,
        batch_size=batch_size,
        overlap_factor=overlap_factor
    )

    # This extraction is not strictly necessary, but it is useful to demonstrate how to extract scalers
    # from the loaders. We could also just use the saved scaler directly.
    # Extract scalers from train loaders in loaders_by_sequence_length
    scalers = {
        sequence_length: [
            train_loader.dataset.dataset.scaler  # Access the scaler from the base dataset
            for train_loader, _ in fold_loaders
        ]
        for sequence_length, fold_loaders in loaders_by_sequence_length.items()
    }

    # Setup scalers
    for seq_len, fold_scalers in scalers.items():
        print(f"Sequence Length: {seq_len}")
        for fold_idx, scaler in enumerate(fold_scalers, start=1):
            print(f"  Fold {fold_idx}: {scaler}")    

    # Prepare test loaders for each sequence length
    test_loaders_by_sequence_length = {}

    for sequence_length, fold_scalers in scalers.items():
        print(f"Preparing test loader for sequence length: {sequence_length}")

        # Create test dataset using the first scaler from the folds (or loop through all folds if needed)
        # Assuming that tabular_dataset_test is already initialized
        test_dataset = CMAPSSDataset(
            data_dir=data_dir,
            data_index=data_index,
            feature_cols=feature_cols,
            mode="test",
            compute_engineered_features=compute_engineered_features,
            features_to_engineer=["sensor_12", "sensor_7", "sensor_21", "sensor_20", "sensor_11", "sensor_4", "sensor_15"],
            scaler=fold_scalers[0]  # Use the scaler from the first fold as a representative
        )

        # Create the DataLoader for this sequence length
        test_loader, _ = prepare_sequence_dataloader(
            tabular_dataset=test_dataset,
            batch_size=batch_size,
            mode="test",
            sequence_length=sequence_length,
            overlap=int(sequence_length * overlap_factor)
        )

        # Store the DataLoader
        test_loaders_by_sequence_length[sequence_length] = test_loader

    # Verify the test loaders
    for seq_len, test_loader in test_loaders_by_sequence_length.items():
        print(f"Test loader for sequence length {seq_len}: {len(test_loader.dataset)} sequences")

    X_val, y_val_first, y_val_last, engine_ids = gather_base_model_predictions_first_and_last(
        loaders_by_sequence_length=loaders_by_sequence_length,
        sequence_lengths=[30, 60, 90, 120],
        save_path="saved_weights",
        device="cuda"
    )

    # Inspect shapes
    print("X_val shape:", X_val.shape)  # e.g. (num_engines, 8) if 4 seq_len models
    print("First RUL shape:", y_val_first.shape)
    print("Last RUL shape:", y_val_last.shape)
    print("Engine IDs:", engine_ids)

    print("Done")