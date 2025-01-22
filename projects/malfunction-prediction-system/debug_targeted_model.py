import os
import pandas as pd
import torch
from codebase.v7.models import TransformerRULModel
from codebase.v8.data import prepare_kfold_cross_validation_loaders
from codebase.v8.dataset import CMAPSSDataset
import numpy as np
from sklearn.metrics import mean_squared_error

# Save the model state dictionary
save_path = "saved_weights/seq-length-specific-models"
feature_cols_path = os.path.join(save_path, "feature_cols.pth")

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

# Group	Number of Engines	Sequence Lengths for Training
#   Short 7	                30, 60
#   Medium 76	            60, 90, 120
#   Long 17	                120, 150

# Example usage
overlap_factor = 0.5
sequence_lengths = [30, 60, 90, 120, 150]
n_splits = 5
batch_size = 256

# Calculate maximum cycles per engine
max_cycles_per_engine = train_data.groupby('engine_id')['cycle'].max()
fold_loaders_per_sequence_length = {}

for seq_len in sequence_lengths:
    print(f"\n--- Create dataset and loaders for Sequence Length: {seq_len} ---")
    
    scaler_path = os.path.join(save_path, f"scaler_{seq_len}.pth")

    # Filter engines with sufficient cycles for the current sequence length
    valid_engine_ids = max_cycles_per_engine[max_cycles_per_engine >= seq_len].index
    print(f"  Number of engines with at least {seq_len} cycles: {len(valid_engine_ids)}")

    filter_fn = lambda df: df[df['engine_id'].isin(
        df.groupby('engine_id')['cycle'].max()[lambda x: x >= seq_len].index
    )]    
    
    # Create a filtered tabular dataset
    filtered_tabular_dataset = CMAPSSDataset(
        data_dir=data_dir,
        data_index=data_index,
        feature_cols=feature_cols,
        mode="train",
        compute_engineered_features=compute_engineered_features,
        features_to_engineer=["sensor_12", "sensor_7", "sensor_21", "sensor_20", "sensor_11", "sensor_4", "sensor_15"],        
        filter_fn=filter_fn
    )

    # save the scaler (all derived datasets use this scaler)
    torch.save(filtered_tabular_dataset.scaler, scaler_path)
    
    fold_loaders = prepare_kfold_cross_validation_loaders(
            original_dataset=filtered_tabular_dataset,
            sequence_length=seq_len,
            overlap=int(overlap_factor * seq_len),
            n_splits=n_splits,
            batch_size=batch_size
        )
    
    fold_loaders_per_sequence_length[seq_len] = fold_loaders, filtered_tabular_dataset

scalers = {
    sequence_length: [
        train_loader.dataset.dataset.scaler  # Access the scaler from the base dataset
        for train_loader, _ in fold_loaders
    ]
    for sequence_length, (fold_loaders, _) in fold_loaders_per_sequence_length.items()
}

# Setup scalers
for seq_len, fold_scalers in scalers.items():
    print(f"Sequence Length: {seq_len}")
    for fold_idx, scaler in enumerate(fold_scalers, start=1):
        print(f"  Fold {fold_idx}: {scaler}")

from codebase.v8.data import prepare_sequence_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def evaluate_test_set(test_loaders_by_sequence_length, sequence_groups, rul_file_path, get_model_fn, save_path, device='cpu'):
    """
    Evaluate the test set using sequence-length-specific models and weighted average predictions within groups.

    Args:
        test_loaders_by_sequence_length (dict): Test loaders grouped by sequence length.
        sequence_groups (dict): Groups of engines with their corresponding sequence lengths for training.
        rul_file_path (str): Path to the ground truth RUL file.
        get_model_fn (callable): Function to load a model for a given sequence length.
        save_path (str): Path where model weights and configurations are stored.
        device (str): Device for model evaluation ("cpu" or "cuda").

    Returns:
        float: RMSE for the entire test set.
    """
    # Load ground truth RUL values
    ground_truth_rul = np.loadtxt(rul_file_path)
    
    # Map engine IDs to their ground truth RUL
    engine_rul_map = {i + 1: rul for i, rul in enumerate(ground_truth_rul)}

    # Store predictions for each engine
    engine_predictions = {engine_id: [] for engine_id in engine_rul_map.keys()}

    # Store weighted predictions per group, engine, and sequence length
    group_model_predictions = {
        group_name: {engine_id: {} for engine_id in group_engine_ids}
        for group_name, (group_engine_ids, _) in sequence_groups.items()
    }

    # Iterate through groups
    for group_name, (group_engine_ids, sequence_lengths) in sequence_groups.items():
        print(f"Processing Group: {group_name}")
        
        group_weights = []

        # Iterate through sequence lengths for this group
        for seq_len in sequence_lengths:
            test_loader = test_loaders_by_sequence_length[seq_len]

            # Load the appropriate model
            model_path = os.path.join(save_path, f"model_seq_len_{seq_len}.pth")
            model, model_config = get_model_fn(seq_len, input_dim=len(test_loader.dataset.feature_cols))
            
            # Load the trained weights
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            # Get the model's weight based on its validation RMSE
            weight = 1 / model_config["val_rmse"]
            group_weights.append(weight)

            # Iterate through the test loader for this sequence length
            for features, mask, engine_ids, _ in test_loader:
                features, mask = features.to(device), mask.to(device)
                outputs = model(features, mask).cpu().detach().numpy()

                # Map predictions to engine IDs
                for i, engine_id in enumerate(engine_ids):
                    engine_id = int(engine_id.item())

                    if engine_id in group_engine_ids:
                        # Only keep the last prediction (latest sequence) for each model
                        group_model_predictions[group_name][engine_id][seq_len] = outputs[i]

        # Combine predictions for each engine in the group
        group_weights = np.array(group_weights) / np.sum(group_weights)  # Normalize weights
        
        for engine_id, predictions_by_model in group_model_predictions[group_name].items():
            # Ensure predictions align with the group weights
            predictions = [predictions_by_model[seq_len] for seq_len in sequence_lengths if seq_len in predictions_by_model]
            predictions = np.array(predictions)

            # Weighted average of predictions across sequence lengths
            if len(predictions) > 0:
                engine_predictions[engine_id].append(
                    np.average(predictions, axis=0, weights=group_weights)
                )

    # Compute RMSE across all engines
    all_predictions = []
    all_ground_truth = []
    for engine_id, predictions in engine_predictions.items():
        if predictions:  # Skip engines without predictions
            avg_prediction = np.mean(predictions)  # Average predictions for the engine
            all_predictions.append(avg_prediction)
            all_ground_truth.append(engine_rul_map[engine_id])

    rmse = np.sqrt(mean_squared_error(all_ground_truth, all_predictions))
    print(f"Test Set RMSE: {rmse}")
    return rmse

def get_model(sequence_length, input_dim):
    """
    Create or load a new model instance for a specific sequence length.
    You can adapt this function to your own model initialization needs.
    """
    assert sequence_length in [30, 60, 90, 120, 150], f"Invalid sequence length: {sequence_length}"

    # input_dim = len(tabular_dataset.feature_cols)

    # Load the model config from disk (must match training config)
    model_config = torch.load(
        os.path.join(save_path, f"model_config_seq_len_{sequence_length}.pth")
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

import pandas as pd

def determine_engine_ids_by_range(train_data, short_max=150, medium_max=250):
    """
    Determines the engine IDs that fall into Short, Medium, and Long groups based on max cycles.

    Args:
        train_data (pd.DataFrame): Training data containing 'engine_id' and 'cycle' columns.
        short_max (int): Maximum cycles for the Short group.
        medium_max (int): Maximum cycles for the Medium group (must be > short_max).

    Returns:
        dict: A dictionary containing engine IDs for each group.
    """
    # Calculate max cycles per engine
    max_cycles_per_engine = train_data.groupby('engine_id')['cycle'].max()

    # Assign groups based on max cycles
    short_engines = max_cycles_per_engine[max_cycles_per_engine <= short_max].index.tolist()
    medium_engines = max_cycles_per_engine[(max_cycles_per_engine > short_max) & (max_cycles_per_engine <= medium_max)].index.tolist()
    long_engines = max_cycles_per_engine[max_cycles_per_engine > medium_max].index.tolist()

    return {
        "Short": short_engines,
        "Medium": medium_engines,
        "Long": long_engines
}

# Load training data
test_data = pd.read_csv("./data/CMAPSSData/test_FD001.txt", sep=r'\s+', header=None)
test_data.columns = ['engine_id', 'cycle'] + [f'os_{i+1}' for i in range(3)] + [f'sensor_{i}' for i in range(1, 22)]

# Determine engine groups
engine_groups = determine_engine_ids_by_range(test_data, short_max=150, medium_max=250)

# Define sequence lengths for each group
sequence_lengths_by_group = {
    "Short": [30, 60],
    "Medium": [60, 90, 120],
    "Long": [120, 150]
}

# Create the sequence_groups variable
sequence_groups = {
    group: (engine_ids, sequence_lengths_by_group[group])
    for group, engine_ids in engine_groups.items()
}

# Print the resulting sequence_groups for verification
for group, (engine_ids, seq_lengths) in sequence_groups.items():
    print(f"{group} Group: {len(engine_ids)} engines, Sequence Lengths: {seq_lengths}")


if __name__ == "__main__":

    # Example usage
    rmse = evaluate_test_set(
        test_loaders_by_sequence_length=test_loaders_by_sequence_length,
        sequence_groups=sequence_groups,
        rul_file_path="data/CMAPSSData/RUL_FD001.txt",
        get_model_fn=get_model,
        save_path=save_path
    )