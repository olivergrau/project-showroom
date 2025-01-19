import torch
from codebase.v6.data import prepare_sequence_dataloader
import os
import pandas as pd
import torch
from codebase.v6.dataset import CMAPSSDataset
from codebase.v7.data import prepare_loaders_for_sequence_lengths
from codebase.v7.prediction import get_model

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

# Prepare test loaders for each sequence length
test_loaders_by_sequence_length = {}

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

scalers = {
    sequence_length: [
        train_loader.dataset.dataset.scaler  # Access the scaler from the base dataset
        for train_loader, _ in fold_loaders
    ]
    for sequence_length, fold_loaders in loaders_by_sequence_length.items()
}

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
    
def predict_rul_for_engine(model, test_loader, engine_id, device="cpu"):
    """
    Predict the RUL for a single engine by first locating its data in the test loader
    and then performing a prediction on the relevant sequences.

    Args:
        model (nn.Module): Trained RUL prediction model.
        test_loader (DataLoader): Test DataLoader, providing sequences for all engines.
        engine_id (int): The ID of the engine to predict RUL for (1-based ID).
        device (str): Device to run the evaluation on ("cpu" or "cuda").

    Returns:
        float: Predicted RUL for the specified engine.
    """
    model.eval()

    # Variables to store the data for the specified engine
    engine_features = []
    engine_masks = []
    engine_cycles = []

    # Step 1: Locate data for the specified engine in the test loader
    with torch.no_grad():
        for (features, masks, engine_ids, cycles) in test_loader:
            for i in range(len(engine_ids)):
                e_id = int(engine_ids[i].item())  # Engine ID (1-based in dataset)
                if e_id == engine_id:
                    engine_features.append(features[i])
                    engine_masks.append(masks[i])
                    engine_cycles.append(cycles[i])

    # If no data was found for the specified engine
    if not engine_features:
        raise ValueError(f"No data found for engine ID {engine_id} in the test loader.")

    # Step 2: Stack the data for prediction
    engine_features = torch.stack(engine_features).to(device)  # Shape: (num_sequences, seq_len, num_features)
    engine_masks = torch.stack(engine_masks).to(device)        # Shape: (num_sequences, seq_len)
    engine_cycles = torch.stack(engine_cycles).cpu().numpy()   # Shape: (num_sequences, seq_len)

    # Step 3: Predict RUL using only the sequences for the specified engine
    best_cycle = -float('inf')
    best_pred = None

    print(f"Feeding the model with {len(engine_features)} sequences for engine {engine_id}...")

    with torch.no_grad():
        outputs = model(engine_features, engine_masks)  # Shape: (num_sequences,)

        for i in range(len(engine_features)):
            max_cycle_in_window = engine_cycles[i].max()  # Find the highest cycle in this sequence

            # Update the prediction if this sequence is more "final"
            if max_cycle_in_window > best_cycle:
                best_cycle = max_cycle_in_window
                best_pred = outputs[i].item()

    # Step 4: Return the predicted RUL    
    return best_pred

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 120

    # Load model
    model, _ = get_model(seq_len, tabular_dataset=tabular_dataset, save_path=save_path)
    
    model_ckpt = torch.load(
        f"{save_path}/model_seq_len_{seq_len}.pth",
        map_location=device, weights_only=True
    )
    model.load_state_dict(model_ckpt["model_state_dict"])
    model.to(device)

    # Example: Predict RUL for Engine ID 5
    engine_id_to_predict = 2

    # Assuming `trained_model`, `test_loader`, and `device` are already defined
    rul_prediction = predict_rul_for_engine(model, test_loader, engine_id=engine_id_to_predict, device="cuda")

    print(f"Predicted RUL for Engine {engine_id_to_predict}: {rul_prediction:.4f}")