import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class CMAPSSDataset(Dataset):
    def __init__(self, data_dir, data_index, feature_cols, mode="train", transform=None, random_state=42):
        """
        Initialize the CMAPSSDataset.

        Args:
            data_dir (str): Directory containing the dataset files.
            data_index (int): Index of the dataset file to load.
            mode (str): Mode of the dataset, "train", "val", or "test".
            feature_cols (list): List of feature columns to include in the dataset.
            random_state (int): Random seed for reproducibility.
            transform (ScalerTransform, optional): Transform to normalize the features.
        """
        assert mode in ["train", "test"], "Mode must be 'train', 'val', or 'test'."
        assert data_index in [0, 1, 2, 3], "Data index must be between 0 and 3."

        self.data_index = data_index
        self.mode = mode
        self.data_dir = data_dir
        self.random_state = random_state
        self.transform = transform
        self.feature_cols = feature_cols

        # Load data
        self.data = self._load_data()

        # Apply preprocessing based on mode
        if mode in ["train"]:
            self.data = self._compute_rul(self.data) # for test mode no RUL is provided           
        
        # Apply transformations to the entire dataset during initialization
        self._apply_transforms()

    def _load_data(self):
        """Load the dataset based on the mode."""
        file_name = f"train_FD00{self.data_index}.txt" if self.mode in ["train", "val"] else f"test_FD00{self.data_index}.txt"
        file_path = os.path.join(self.data_dir, file_name)
        column_names = ['engine_id', 'cycle'] + \
                       [f'os_{i+1}' for i in range(3)] + \
                       [f'sensor_{i}' for i in range(1, 22)]
        data = pd.read_csv(file_path, sep=r"\s+", header=None, names=column_names, engine='python')
        data = data.loc[:, ~data.columns.str.contains('Unnamed')]

        # keep only the selected feature columns + engine and cycle
        data = data[['engine_id', 'cycle'] + self.feature_cols]
        return data

    def _compute_rul(self, data):
        """Compute Remaining Useful Life (RUL) for the training dataset."""
        max_cycles = data.groupby('engine_id')['cycle'].max().reset_index()
        max_cycles.columns = ['engine_id', 'max_cycle']
        data = data.merge(max_cycles, on='engine_id')
        data['RUL'] = data['max_cycle'] - data['cycle']
        data = data.drop(columns=['max_cycle'])
        return data

    def _apply_transforms(self):
        """Apply the transform to the entire dataset."""
        if self.mode in ["train"]:
            self.data = self.data[['engine_id', 'cycle'] + self.feature_cols + ['RUL']]
        else:
            self.data = self.data[['engine_id', 'cycle'] + self.feature_cols]
        
        if self.transform:        
            # Apply transformation to the remaining columns
            self.data[self.feature_cols] = self.transform.transform(self.data[self.feature_cols])

    def __len__(self):
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a single sample."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        
        if self.mode in ["train"]:
            target = row['RUL']
            features = row[2:-1].values.astype(float)  # Exclude engine_id, cycle, and RUL
            return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
        else:            
            features = row[2:].values.astype(float)  # Exclude engine_id, cycle, and RUL
            return torch.tensor(features, dtype=torch.float32)