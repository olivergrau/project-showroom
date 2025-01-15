from codebase.v5.data import prepare_kfold_cross_validation_loaders, prepare_sequence_dataloader
import torch
from torch.utils.data import DataLoader
from codebase.v5.dataset import CMAPSSDataset, SequenceCMAPSSDataset
import pandas as pd
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from codebase.v5.models import TransformerRULModel
import os
import torch.nn.functional as F

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

    # Save the model state dictionary
    save_path = "saved_weights/"
    config_path = os.path.join(save_path, "config.pth")
    model_path = os.path.join(save_path, "model.pth")
    scaler_path = os.path.join(save_path, "scaler.pth")
    feature_cols_path = os.path.join(save_path, "feature_cols.pth")

    config = torch.load(config_path)
    feature_cols = torch.load(feature_cols_path)

    # Reinitialize the model architecture
    checkpoint = torch.load(model_path, weights_only=True)

    print(f"Model loaded from {model_path}")
    print(f"Model configuration: {config}")

    # print used overlap, sequence length and batch size
    print(f"Overlap: {checkpoint['overlap']}")
    print(f"Sequence Length: {checkpoint['sequence_length']}")
    print(f"Batch Size: {checkpoint['batch_size']}")

    # Set up paths and features
    data_dir = "data/CMAPSSData"
    data_index = 1  # Choose FD001, FD002, etc.

    compute_engineered_features = True
    sequence_length = checkpoint['sequence_length']
    batch_size = checkpoint['batch_size']
    overlap = checkpoint['overlap']

    # save the scaler for the test dataset
    scaler = torch.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}")

    tabular_dataset_test = CMAPSSDataset(
        data_dir=data_dir,
        data_index=data_index,
        feature_cols=feature_cols,
        mode="test",
        compute_engineered_features=compute_engineered_features,
        features_to_engineer=["sensor_12", "sensor_7", "sensor_21", "sensor_20", "sensor_11", "sensor_4", "sensor_15"],
        scaler=scaler
    )
    
    test_loader, sequence_test = prepare_sequence_dataloader(
        tabular_dataset=tabular_dataset_test, batch_size=batch_size, mode="test", sequence_length=sequence_length, overlap=overlap        
    )

    # Load the RUL file
    rul_file_path = "./data/CMAPSSData/RUL_FD001.txt"  # Adjust the path if needed
    rul_df = pd.read_csv(rul_file_path, header=None, names=["RUL"])

    # RUL values for each engine
    rul_values = rul_df["RUL"].values  # Shape: (num_engines,)

    # RecommendationNet model
    loaded_model = TransformerRULModel(**config).to(device)

    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    # Example: Predict RUL for Engine ID n
    engine_id_to_predict = 1

    # Assuming `trained_model`, `test_loader`, and `device` are already defined
    rul_prediction = predict_rul_for_engine(loaded_model, test_loader, engine_id=engine_id_to_predict, device="cuda")

    print(f"Predicted RUL for Engine {engine_id_to_predict}: {rul_prediction:.4f}")
    