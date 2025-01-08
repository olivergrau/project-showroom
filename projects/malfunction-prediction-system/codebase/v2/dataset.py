import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class CMAPSSDataset(Dataset):
    def __init__(self, data_dir, data_index, feature_cols, mode="train", 
                 transform=None, scaler=None, compute_engineered_features=True, 
                 standard_scale=True, random_state=42):
        """
        Initialize the CMAPSSDataset.

        Args:
            data_dir (str): Directory containing the dataset files.
            data_index (int): Index of the dataset file to load.
            feature_cols (list): List of original feature columns to include in the dataset.
            mode (str): Mode of the dataset, "train" or "test".
            transform (callable, optional): Transformation to apply to the features.
            scaler (StandardScaler, optional): Pre-fitted StandardScaler for test mode.
            random_state (int): Random seed for reproducibility.
        """
        assert mode in ["train", "test"], "Mode must be 'train' or 'test'."
        assert data_index in [0, 1, 2, 3], "Data index must be between 0 and 3."
        assert scaler is not None or mode == "train", "A pre-fitted StandardScaler must be provided in test mode."

        self.standard_scale = standard_scale
        self.data_index = data_index
        self.mode = mode
        self.data_dir = data_dir
        self.random_state = random_state
        self.transform = transform
        self.feature_cols = feature_cols
        self.scaler = scaler

        # Load data
        self.data = self._load_data()

        # Compute engineered features
        if compute_engineered_features:
            self.data = self._compute_engineered_features(self.data)

        # Compute RUL if in train mode
        if self.mode == "train":
            self.data = self._compute_rul(self.data)

        # Scale features
        if self.standard_scale:
            self._scale_features()

        # Apply transformations (optional)
        self._apply_transforms()

    def _load_data(self):
        """Load the dataset based on the mode."""
        file_name = f"train_FD00{self.data_index}.txt" if self.mode == "train" else f"test_FD00{self.data_index}.txt"
        file_path = os.path.join(self.data_dir, file_name)
        column_names = ['engine_id', 'cycle'] + \
                       [f'os_{i+1}' for i in range(3)] + \
                       [f'sensor_{i}' for i in range(1, 22)]
        data = pd.read_csv(file_path, sep=r"\s+", header=None, names=column_names, engine='python')

        # keep only the selected feature columns + engine and cycle
        data = data[['engine_id', 'cycle'] + self.feature_cols]
        return data

    def _compute_engineered_features(self, data):
        """Dynamically compute engineered features for the dataset."""
        feature_data = data.copy()

        for sensor in [col for col in self.feature_cols if col.startswith("sensor_")]:
            # Calculate rolling metrics
            feature_data[f'{sensor}_ma'] = feature_data.groupby('engine_id')[sensor].rolling(5).mean().reset_index(level=0, drop=True)
            feature_data[f'{sensor}_std'] = feature_data.groupby('engine_id')[sensor].rolling(5).std().reset_index(level=0, drop=True)

            # Calculate differences
            feature_data[f'{sensor}_diff'] = feature_data.groupby('engine_id')[sensor].diff()

        # Handle NaN values
        if self.mode == "train":
            feature_data = feature_data.dropna()  # Drop rows with NaN in training mode
        else:
            feature_data.fillna(feature_data.mean(), inplace=True)  # Replace NaN with column mean in test mode

        return feature_data

    def _compute_rul(self, data):
        """Compute Remaining Useful Life (RUL) for the training dataset."""
        max_cycles = data.groupby('engine_id')['cycle'].max().reset_index()
        max_cycles.columns = ['engine_id', 'max_cycle']
        data = data.merge(max_cycles, on='engine_id')
        data['RUL'] = data['max_cycle'] - data['cycle']
        data = data.drop(columns=['max_cycle'])
        return data

    def _scale_features(self):
        """Scale features using StandardScaler."""
        feature_cols = [col for col in self.data.columns if col not in ['engine_id', 'cycle', 'RUL']]

        if self.mode == "train":
            # Instantiate and fit the scaler
            self.scaler = StandardScaler()
            self.data[feature_cols] = self.scaler.fit_transform(self.data[feature_cols])
        elif self.mode == "test":
            assert self.scaler is not None, "A pre-fitted StandardScaler must be provided in test mode."
            self.data[feature_cols] = self.scaler.transform(self.data[feature_cols])

    def _apply_transforms(self):
        """Apply transformations to the dataset."""
        # Collect all features including engineered ones
        all_feature_cols = [col for col in self.data.columns if col not in ['engine_id', 'cycle', 'RUL']]
        self.data = self.data[['engine_id', 'cycle'] + all_feature_cols + (['RUL'] if self.mode == "train" else [])]

        if self.transform:
            self.data[all_feature_cols] = self.transform.transform(self.data[all_feature_cols])

    def __len__(self):
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a single sample."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]

        # Extract features dynamically
        target = row['RUL'] if self.mode == "train" else None
        features = row[2:-1].values.astype(float) if self.mode == "train" else row[2:].values.astype(float)

        if target is not None:
            return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
        
        return torch.tensor(features, dtype=torch.float32)