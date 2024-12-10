import os
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from starter.src.ml.data import process_data
from starter.src.ml.model import train_model, inference, compute_model_metrics, test_slices

# Define paths
model_path = "../model/trained_model.pkl"
encoder_path = "../model/encoder.pkl"
lb_path = "../model/label_binarizer.pkl"

# Load the data
df = pd.read_csv("../data/census_no_spaces.csv")
label_col = "salary"

# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.20)

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Check if the model exists
if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(lb_path):
    print("Loading existing model, encoder, and label binarizer...")
    # Load the existing model, encoder, and label binarizer
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)
else:
    print("No existing model found. Training a new model...")

    # Process training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label_col, training=True
    )

    # Train the model
    model = train_model(X_train, y_train)

    # Save the model, encoder, and label binarizer
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(lb, lb_path)

# Process the test data with the process_data function
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label=label_col, training=False, encoder=encoder, lb=lb
)

# Evaluate the model
preds = inference(model, X_test)
precision, recall, f1 = compute_model_metrics(y_test, preds)
print(f"Overall performance: Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Evaluate slices of the test data
slice_metrics = test_slices(test, model, encoder, lb, cat_features, label_col)

# Print the metrics for each slice
for slice_name, metrics in slice_metrics.items():
    precision, recall, fbeta = metrics
    print(f"Slice: {slice_name}")
    print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {fbeta:.2f}")
