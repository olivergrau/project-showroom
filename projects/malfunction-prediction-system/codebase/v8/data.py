import torch
from torch.utils.data import DataLoader
from codebase.v6.dataset import CMAPSSDataset, SequenceCMAPSSDataset
import pandas as pd
from sklearn.model_selection import GroupKFold
from codebase.v6.models import TransformerRULModel
import os
import torch.nn.functional as F
import torch

def validate_kfold_loaders(original_dataset, fold_loaders):
    """
    Validates the integrity of k-fold DataLoaders against the original dataset.
    Also checks for NaN in features, targets, and cycles for both train and val loaders.

    Args:
        original_dataset (CMAPSSDataset): The original dataset instance.
        fold_loaders (list): List of (train_loader, val_loader) tuples for k-fold cross-validation.

    Returns:
        bool: True if all validations pass, otherwise raises an AssertionError or ValueError.
    """
    original_data = original_dataset.data
    used_indices = set()  # Track all rows covered across folds

    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"Validating Folds: {fold + 1}")
        
        # Get engine IDs for train and validation sets
        train_engines = set(train_loader.dataset.dataset.data['engine_id'])
        val_engines = set(val_loader.dataset.dataset.data['engine_id'])
        
        # Check group exclusivity
        assert train_engines.isdisjoint(val_engines), \
            f"Fold {fold + 1}: Train and validation sets share engine IDs: {train_engines & val_engines}"

        # ========== Validate Training Data ========== #
        for batch_idx, (features, targets, _, engine_ids, cycles) in enumerate(train_loader):
            # features shape: (batch_size, seq_len, num_features)
            # targets shape:  (batch_size,)
            # engine_ids shape: (batch_size,)
            # cycles shape:   (batch_size, seq_len)

            batch_size = features.size(0)
            for idx in range(batch_size):
                seq = features[idx]          # shape: (seq_len, num_features)
                target = targets[idx]        # scalar RUL value
                engine_id = engine_ids[idx]  # scalar engine ID
                seq_cycles = cycles[idx]     # shape: (seq_len,)

                # ---------- NaN checks ----------
                if torch.isnan(seq).any():
                    raise ValueError(
                        f"[Train] NaN found in features for engine {engine_id.item()}, "
                        f"batch {batch_idx}, sample {idx}"
                    )
                if torch.isnan(target):
                    raise ValueError(
                        f"[Train] NaN found in target for engine {engine_id.item()}, "
                        f"batch {batch_idx}, sample {idx}"
                    )
                # Convert cycles to float for NaN check (since cycles are usually int)
                if torch.isnan(seq_cycles.float()).any():
                    raise ValueError(
                        f"[Train] NaN found in cycles for engine {engine_id.item()}, "
                        f"batch {batch_idx}, sample {idx}"
                    )
                # ---------------------------------

                # Identify real (non-padded) timesteps (padded cycles = 0)
                real_timesteps = (seq_cycles != 0).sum().item()

                # Skip if everything is padded (unlikely in normal usage)
                if real_timesteps == 0:
                    continue

                # For each real timestep t in this sequence:
                for t in range(real_timesteps):
                    cycle_val = float(seq_cycles[t].item())
                    original_row = original_data[
                        (original_data['engine_id'] == float(engine_id)) & 
                        (original_data['cycle'] == cycle_val)
                    ]
                    assert not original_row.empty, (
                        f"Missing data for engine {engine_id}, cycle {cycle_val} in training set"
                    )

                    # Check feature alignment
                    original_features = original_row.iloc[0, 2:-1].values  # columns 2..-1 are features
                    assert torch.allclose(
                        seq[t], torch.tensor(original_features, dtype=torch.float32), atol=1e-6
                    ), f"Mismatch in features for engine {engine_id}, cycle {cycle_val}"

                # Verify RUL target using the last real cycle
                last_cycle = float(seq_cycles[real_timesteps - 1].item())
                expected_rul = original_data[
                    (original_data['engine_id'] == float(engine_id)) &
                    (original_data['cycle'] == last_cycle)
                ]['RUL'].iloc[0]

                assert torch.isclose(
                    target, torch.tensor(expected_rul, dtype=torch.float32), atol=1e-6
                ), f"Mismatch in RUL for engine {engine_id}, last cycle {last_cycle}"

        # ========== Validate Validation Data (same logic) ========== #
        for batch_idx, (features, targets, _, engine_ids, cycles) in enumerate(val_loader):
            batch_size = features.size(0)
            for idx in range(batch_size):
                seq = features[idx]
                target = targets[idx]
                engine_id = engine_ids[idx]
                seq_cycles = cycles[idx]

                # ---------- NaN checks ----------
                if torch.isnan(seq).any():
                    raise ValueError(
                        f"[Val] NaN found in features for engine {engine_id.item()}, "
                        f"batch {batch_idx}, sample {idx}"
                    )
                if torch.isnan(target):
                    raise ValueError(
                        f"[Val] NaN found in target for engine {engine_id.item()}, "
                        f"batch {batch_idx}, sample {idx}"
                    )
                if torch.isnan(seq_cycles.float()).any():
                    raise ValueError(
                        f"[Val] NaN found in cycles for engine {engine_id.item()}, "
                        f"batch {batch_idx}, sample {idx}"
                    )
                # ---------------------------------

                real_timesteps = (seq_cycles != 0).sum().item()
                if real_timesteps == 0:
                    continue

                for t in range(real_timesteps):
                    cycle_val = float(seq_cycles[t].item())
                    original_row = original_data[
                        (original_data['engine_id'] == float(engine_id)) & 
                        (original_data['cycle'] == cycle_val)
                    ]
                    assert not original_row.empty, (
                        f"Missing data for engine {engine_id}, cycle {cycle_val} in validation set"
                    )

                    original_features = original_row.iloc[0, 2:-1].values
                    assert torch.allclose(
                        seq[t], torch.tensor(original_features, dtype=torch.float32), atol=1e-6
                    ), f"Mismatch in features for engine {engine_id}, cycle {cycle_val}"

                last_cycle = float(seq_cycles[real_timesteps - 1].item())
                expected_rul = original_data[
                    (original_data['engine_id'] == float(engine_id)) &
                    (original_data['cycle'] == last_cycle)
                ]['RUL'].iloc[0]
                assert torch.isclose(
                    target, torch.tensor(expected_rul, dtype=torch.float32), atol=1e-6
                ), f"Mismatch in RUL for engine {engine_id}, last cycle {last_cycle}"

        # Track covered indices
        train_indices = train_loader.dataset.dataset.data.index
        val_indices = val_loader.dataset.dataset.data.index
        used_indices.update(train_indices)
        used_indices.update(val_indices)

    # Finally, ensure overall coverage if that's desired
    assert len(used_indices) == len(original_data), \
        "Not all rows in the original dataset are covered by the k-fold loaders"

    print("All validations passed!")
    return True

def prepare_loaders_for_sequence_lengths(
    sequence_lengths, 
    tabular_dataset, 
    n_splits, 
    batch_size, 
    overlap_factor=0.5
):
    """
    Prepare DataLoaders for train and validation for the specified sequence lengths.

    Args:
        sequence_lengths (list): List of desired sequence lengths.
        tabular_dataset (CMAPSSDataset): The original tabular dataset.
        n_splits (int): Number of folds for cross-validation.
        batch_size (int): Batch size for the DataLoaders.
        overlap_factor (float): Overlap factor for sequences (e.g., 0.5 for 50% overlap).

    Returns:
        dict: A dictionary containing fold loaders for each sequence length.
              Format: {sequence_length: [(train_loader, val_loader), ...]}
    """
    loaders_by_sequence_length = {}

    for seq_len in sequence_lengths:
        print(f"\nPreparing loaders for sequence length: {seq_len}")
        overlap = int(seq_len * overlap_factor)
        
        # Generate fold loaders for the current sequence length
        fold_loaders = prepare_kfold_cross_validation_loaders(
            original_dataset=tabular_dataset,
            sequence_length=seq_len,
            overlap=overlap,
            n_splits=n_splits,
            batch_size=batch_size
        )
        
        loaders_by_sequence_length[seq_len] = fold_loaders

        # # Display information about the first fold for debugging
        # train_loader, val_loader = fold_loaders[0]
        # print(f"  Number of training sequences: {len(train_loader.dataset)}")
        # print(f"  Number of validation sequences: {len(val_loader.dataset)}")
        
        # first_train_item = next(iter(train_loader))
        # train_features, train_targets, train_masks, train_engine_ids, cycles = first_train_item
        # print(f"  Training Features Shape: {train_features.shape}")
        # print(f"  Training Targets Shape: {train_targets.shape}")

        # first_val_item = next(iter(val_loader))
        # val_features, val_targets, val_masks, val_engine_ids, cycles = first_val_item
        # print(f"  Validation Features Shape: {val_features.shape}")
        # print(f"  Validation Targets Shape: {val_targets.shape}")
    
    return loaders_by_sequence_length

# Function to implement Group k-Fold with your custom dataset
def prepare_kfold_cross_validation_loaders(original_dataset, sequence_length, overlap, n_splits=5, batch_size=32):
    """
    Implements Group k-Fold Cross-Validation for the CMAPSS dataset.
    
    Args:
        original_dataset (CMAPSSDataset): The original dataset instance.
        sequence_length (int): Length of each sequence.
        overlap (int): Overlap between sequences.
        n_splits (int): Number of folds for cross-validation.
        batch_size (int): Batch size for the DataLoader.
    
    Returns:
        List[Tuple[DataLoader, DataLoader]]: Training and validation DataLoaders for each fold.
    """

    # Extract engine IDs and features for grouping
    engine_ids = original_dataset.data['engine_id'].values
    features = original_dataset.data[original_dataset.feature_cols].values
    targets = original_dataset.data['RUL'].values

    # Initialize GroupKFold
    group_kfold = GroupKFold(n_splits=n_splits)

    fold_loaders = []  # Store DataLoaders for each fold

    # Perform Group k-Fold split (features and targets are not directly used for the split)
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(features, targets, groups=engine_ids)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Create train and validation datasets
        train_data = original_dataset.data.iloc[train_idx]
        val_data = original_dataset.data.iloc[val_idx]

        train_dataset = SequenceCMAPSSDataset(
            dataset=CMAPSSDatasetSubset(train_data, original_dataset.feature_cols, mode="train", scaler=original_dataset.scaler),
            sequence_length=sequence_length,
            overlap=overlap
        )

        val_dataset = SequenceCMAPSSDataset(
            dataset=CMAPSSDatasetSubset(val_data, original_dataset.feature_cols, mode="train", scaler=original_dataset.scaler),
            sequence_length=sequence_length,
            overlap=overlap
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=lambda batch: collate_fn(batch, mode="train"))
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=lambda batch: collate_fn(batch, mode="train"))

        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders

# Helper class to create a subset of the CMAPSSDataset
class CMAPSSDatasetSubset(CMAPSSDataset):
    def __init__(self, data_subset, feature_cols, mode, scaler=None):
        """
        Creates a subset of the CMAPSSDataset using the provided data.

        Args:
            data_subset: Subset of the original dataset data.
            feature_cols (list): List of feature columns to include.
            mode (str): Mode of the dataset, "train" or "test".
        """
        self.data = data_subset
        self.feature_cols = feature_cols
        self.mode = mode
        self.scaler = scaler

def collate_fn(batch, mode="train"):
        """
        Custom collate function to handle padding and masking.
        
        Args:
            batch (list): List of tuples:
                - Train mode: (features, target, engine_id, cycles)
                - Test mode:  (features, engine_id, cycles)
            mode (str): "train" or "test"
            
        Returns:
            tuple:
                If mode="train": (features_tensor, targets_tensor, masks_tensor, engine_ids, cycles_tensor)
                If mode="test":  (features_tensor, masks_tensor, engine_ids, cycles_tensor)
        """
        features_list = []
        targets_list = []
        masks_list = []
        engine_ids = []
        cycles_list = []
        
        for sample in batch:
            # 1) Handle difference between train & test samples
            if mode == "train":
                features, target, engine_id, cycles = sample
            else:
                features, engine_id, cycles = sample
                target = None  # No target in test mode
            
            # 2) Build a mask to distinguish padded vs. non-padded timesteps
            #    (assuming padded rows are all zeros)
            mask = (features.sum(dim=-1) == 0)  # True for padded positions, False for valid ones
            
            # 3) Skip fully-padded sequences: mask.sum() == len(mask)
            #    means no real timesteps (all zeros).
            if mask.sum() == mask.size(0):  # mask.size(0) is the sequence length
                print("This shouldn't occur: Skipping fully-padded sequence.")
                continue
            
            # Otherwise, keep this sample
            features_list.append(features)
            masks_list.append(mask)
            engine_ids.append(engine_id)
            cycles_list.append(cycles)

            # Only append target in train mode
            if mode == "train":
                targets_list.append(target)

        # 4) Stack features and masks
        features_tensor = torch.stack(features_list)  # (batch_size, seq_len, num_features)
        masks_tensor = torch.stack(masks_list)        # (batch_size, seq_len)
        cycles_tensor = torch.stack(cycles_list)      # (batch_size, seq_len)

        # 5) Return depends on mode
        if mode == "train":
            targets_tensor = torch.stack(targets_list)  # (batch_size,)
            return features_tensor, targets_tensor, masks_tensor, engine_ids, cycles_tensor
        else:
            return features_tensor, masks_tensor, engine_ids, cycles_tensor

def prepare_sequence_dataloader(
        tabular_dataset, batch_size=32, 
        mode="train", sequence_length=30, overlap=0):
    """
    Prepares the DataLoader for the CMAPSS dataset.
    
    Args:
        data_dir (str): Directory containing the dataset files.
        data_index (int): Index of the dataset file to load (0 to 3).
        feature_cols (list): List of feature columns to use.
        batch_size (int): Batch size for the DataLoader.
        mode (str): Mode of the dataset, "train" or "test".
        compute_engineered_features (bool): Whether to include engineered features.
        sequence_length (int): Length of the sequence to extract.
        overlap (int): Number of overlapping time steps between consecutive sequences.
        
    Returns:
        DataLoader: PyTorch DataLoader for the CMAPSS dataset with masking.
    """
    # Wrap it with the sequence-based dataset
    sequence_dataset = SequenceCMAPSSDataset(
        dataset=tabular_dataset, sequence_length=sequence_length, overlap=overlap)
    
    # Create DataLoader with the custom collate function
    dataloader = DataLoader(
        sequence_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=lambda batch: collate_fn(batch, mode=mode)
    )
    
    return dataloader, sequence_dataset

def test_sequence_dataset(dataloader, mode="train"):
    """
    Test the SequenceCMAPSSDataset and display an example output.

    Args:
        dataset (CMAPSSDataset): The original CMAPSSDataset instance.
        sequence_length (int): Length of each sequence.
        batch_size (int): Batch size for the DataLoader.
    """

    # Fetch one batch
    for batch_idx, batch in enumerate(dataloader):
        if mode == "train":
            sequences, targets, masks, engine_ids, cycles = batch
            print(f"Batch {batch_idx + 1}")
            print(f"Sequences Shape: {sequences.shape}")  # Expected: (batch_size, sequence_length, num_features)
            print(f"Targets Shape: {targets.shape}")  # Expected: (batch_size,)
            print(f"Masks Shape: {masks.shape}")  # Expected: (batch_size, sequence_length)
            print(f"Engine Ids Shape: {len(engine_ids)}")
            print(f"Cycles Shape: {cycles.shape}")
            print()
            print(f"Sample Sequence (First in Batch):\n{sequences[0]}")
            print(f"Corresponding Target (First in Batch):\n{targets[0]}")
            print(f"Engine ID (First in batch): {engine_ids[0]}")
            print(f"Cycles (First in batch): {cycles[0]}")
        else:
            sequences, masks, engine_ids, cycles = batch
            print(f"Batch {batch_idx + 1}")
            print(f"Sequences Shape: {sequences.shape}")  # Expected: (batch_size, sequence_length, num_features)
            print(f"Masks Shape: {masks.shape}")  # Expected: (batch_size, sequence_length)
            print(f"Engine Ids Shape: {len(engine_ids)}")
            print(f"Cycles Shape: {cycles.shape}")
            print()
            print(f"Sample Sequence (First in Batch):\n{sequences[0]}")
            print(f"Engine ID (First in batch): {engine_ids[0]}")
            print(f"Cycles (First in batch): {cycles[0]}")
        break  # Only fetch the first batch for demonstration

