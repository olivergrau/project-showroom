import pytest
import pandas as pd
from starter.src.ml.data import process_data
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score
from starter.src.ml.model import compute_model_metrics


def test_process_data_training():
    """Test process_data when training=True."""
    # Sample input data
    data = pd.DataFrame({
        "age": [25, 35],
        "workclass": ["Private", "Self-emp-not-inc"],
        "education": ["Bachelors", "HS-grad"],
        "education-num": [13, 9],
        "salary": [">50K", "<=50K"],
    })
    categorical_features = ["workclass", "education"]

    # Call process_data in training mode
    X, y, encoder, lb = process_data(
        data, categorical_features=categorical_features, label="salary", training=True
    )

    # Assertions
    assert X.shape[1] > len(categorical_features)  # Ensure features are expanded
    assert len(y) == len(data)  # Ensure labels match input size
    assert encoder is not None
    assert lb is not None


def test_process_data_inference_error():
    """Test process_data raises an error when training=False and no encoder/lb is given."""
    # Sample input data
    data = pd.DataFrame({
        "age": [25, 35],
        "workclass": ["Private", "Self-emp-not-inc"],
        "education": ["Bachelors", "HS-grad"],
        "education-num": [13, 9],
        "salary": [">50K", "<=50K"],
    })
    categorical_features = ["workclass", "education"]

    # Expect ValueError when no encoder or lb is provided
    with pytest.raises(ValueError, match="When training=False, both 'encoder' and 'lb' must be provided."):
        process_data(
            data, categorical_features=categorical_features, label="salary", training=False
        )


def test_compute_model_metrics():
    """Test compute_model_metrics function."""
    # Example inputs
    y_true = np.array([1, 0, 1, 1, 0, 1])  # True labels
    y_pred = np.array([1, 0, 1, 0, 0, 1])  # Predicted labels

    # Expected outputs
    expected_precision = precision_score(y_true, y_pred, zero_division=1)
    expected_recall = recall_score(y_true, y_pred, zero_division=1)
    expected_f1 = fbeta_score(y_true, y_pred, beta=1, zero_division=1)

    # Call the function
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    # Assertions
    assert precision == expected_precision, f"Expected {expected_precision}, got {precision}"
    assert recall == expected_recall, f"Expected {expected_recall}, got {recall}"
    assert f1 == expected_f1, f"Expected {expected_f1}, got {f1}"


def test_compute_model_metrics_perfect_predictions():
    """Test compute_model_metrics with perfect predictions."""
    # Perfect predictions
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    # Expected metrics
    expected_precision = 1.0
    expected_recall = 1.0
    expected_f1 = 1.0

    # Call the function
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    # Assertions
    assert precision == expected_precision, f"Expected {expected_precision}, got {precision}"
    assert recall == expected_recall, f"Expected {expected_recall}, got {recall}"
    assert f1 == expected_f1, f"Expected {expected_f1}, got {f1}"


def test_compute_model_metrics_no_true_positives():
    """Test compute_model_metrics with no true positives."""
    # No true positives
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0])

    # Expected metrics
    expected_precision = 1.0  # Defined as 1.0 by zero_division=1
    expected_recall = 0.0
    expected_f1 = 0.0

    # Call the function
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    # Assertions
    assert precision == expected_precision, f"Expected {expected_precision}, got {precision}"
    assert recall == expected_recall, f"Expected {expected_recall}, got {recall}"
    assert f1 == expected_f1, f"Expected {expected_f1}, got {f1}"
