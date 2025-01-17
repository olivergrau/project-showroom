import torch
from torch.utils.data import DataLoader
from codebase.v5.dataset import CMAPSSDataset, SequenceCMAPSSDataset
import pandas as pd
from sklearn.model_selection import GroupKFold
from codebase.v5.models import TransformerRULModel
import os
import torch.nn.functional as F
import torch

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
        Scaler: Scaler used for the dataset.
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
            dataset=CMAPSSDatasetSubset(train_data, original_dataset.feature_cols, mode="train"),
            sequence_length=sequence_length,
            overlap=overlap
        )

        val_dataset = SequenceCMAPSSDataset(
            dataset=CMAPSSDatasetSubset(val_data, original_dataset.feature_cols, mode="train"),
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
    def __init__(self, data_subset, feature_cols, mode):
        """
        Creates a subset of the CMAPSSDataset using the provided data.

        Args:
            data_subset (pd.DataFrame): Subset of the original dataset.
            feature_cols (list): List of feature columns to include.
            mode (str): Mode of the dataset, "train" or "test".
        """
        self.data = data_subset
        self.feature_cols = feature_cols
        self.mode = mode

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

