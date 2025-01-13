import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class CMAPSSDataset(Dataset):
    def __init__(self, data_dir, data_index, feature_cols, mode="train", 
                 transform=None, scaler=None, compute_engineered_features=True, 
                 features_to_engineer=["sensor_12", "sensor_7", "sensor_21", "sensor_20", "sensor_11", "sensor_4", "sensor_15"],
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

        self.features_to_engineer = features_to_engineer
        self.standard_scale = standard_scale
        self.data_index = data_index
        self.mode = mode
        self.data_dir = data_dir
        self.random_state = random_state
        self.transform = transform
        self.feature_cols = feature_cols.copy()
        self.scaler = scaler

        # Load data
        self.data = self._load_data()

        # Compute engineered features
        if compute_engineered_features:
            feature_data, feature_cols = self._compute_engineered_features(self.data)
            self.data = feature_data
            self.feature_cols.extend(feature_cols)

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
        """Dynamically compute engineered features for the dataset and update feature_cols."""
        feature_data = data.copy()
        engineered_features = []  # To track new engineered features

        for sensor in [col for col in self.feature_cols if col.startswith("sensor_") and col in self.features_to_engineer]:
            # Calculate rolling metrics
            ma_feature = f'{sensor}_ma'
            std_feature = f'{sensor}_std'
            diff_feature = f'{sensor}_diff'

            feature_data[ma_feature] = feature_data.groupby('engine_id')[sensor].rolling(5).mean().reset_index(level=0, drop=True)
            feature_data[std_feature] = feature_data.groupby('engine_id')[sensor].rolling(5).std().reset_index(level=0, drop=True)

            # Calculate differences
            feature_data[diff_feature] = feature_data.groupby('engine_id')[sensor].diff()

            # Add the names of engineered features to the list
            engineered_features.extend([ma_feature, std_feature, diff_feature])

        # Handle NaN values
        if self.mode == "train":
            feature_data = feature_data.dropna()  # Drop rows with NaN in training mode
        else:
            feature_data.fillna(feature_data.mean(), inplace=True)  # Replace NaN with column mean in test mode

        return feature_data, engineered_features

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

class SequenceCMAPSSDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sequence_length=30, overlap=0):
        """
        Wraps the CMAPSSDataset to return sequences for each engine.

        Args:
            dataset (Dataset): The original CMAPSSDataset instance.
            sequence_length (int): Number of time steps in each sequence.
            overlap (int): Number of overlapping time steps between consecutive sequences.
        """
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.feature_cols = dataset.feature_cols
        
        # Generate sequence indices per engine
        self.indices = self._generate_indices()

    def _generate_indices(self):
        """
        Generate sequence start and end indices for all engines.
        Ensures no sequence spans across engines and handles overlaps.

        Returns:
            list of tuples: Each tuple contains (engine_id, start_idx, end_idx).
        """
        indices = []
        for engine_id in self.dataset.data['engine_id'].unique():
            engine_data = self.dataset.data[self.dataset.data['engine_id'] == engine_id]
            num_cycles = len(engine_data)
            
            # Generate start indices with the specified overlap
            for start_idx in range(0, num_cycles - self.sequence_length + 1, self.sequence_length - self.overlap):
                end_idx = start_idx + self.sequence_length
                indices.append((engine_id, start_idx, end_idx))
        
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Fetch a sequence and its corresponding RUL (for the last cycle in the sequence).

        Args:
            idx (int): Index of the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]: Sequence features, RUL target, engine_id, and cycles.
        """
        engine_id, start_idx, end_idx = self.indices[idx]
        engine_data = self.dataset.data[self.dataset.data['engine_id'] == engine_id].iloc[start_idx:end_idx]
        
        # Extract features
        features = engine_data[self.feature_cols].values
        features = torch.tensor(features, dtype=torch.float32)
        
        # Extract cycles
        cycles = engine_data['cycle'].values
        cycles = torch.tensor(cycles, dtype=torch.int32)

        # Zero-pad if the sequence is shorter than sequence_length
        if len(features) < self.sequence_length:
            padding = torch.zeros((self.sequence_length - len(features), features.shape[1]))
            features = torch.cat([features, padding], dim=0)
            cycle_padding = torch.zeros(self.sequence_length - len(cycles), dtype=torch.int32)
            cycles = torch.cat([cycles, cycle_padding], dim=0)
        
        # Extract RUL for the last time step
        if self.dataset.mode == "train":
            rul = engine_data['RUL'].iloc[-1]  # RUL for the last cycle
            rul = torch.tensor(rul, dtype=torch.float32)
            return features, rul, engine_id, cycles
        else:
            return features, engine_id, cycles