import pytest
import json
from app import app

@pytest.fixture
def client():
    """
    Set up a test client for the Flask app.
    """
    with app.test_client() as client:
        yield client


def test_prediction_endpoint_with_valid_path(client, tmp_path):
    """
    Test the /prediction endpoint with a valid test dataset path.
    """
    # Create a temporary CSV file for the test
    test_data_path = tmp_path / "testdata.csv"
    test_data_path.write_text("lastmonth_activity,lastyear_activity,number_of_employees,exited\n"
                              "10,20,300,0\n"
                              "5,10,150,1\n")

    # Prepare the request payload
    payload = {
        "test_data_path": str(test_data_path)
    }

    # Send a POST request to the /prediction endpoint
    response = client.post(
        "/prediction",
        data=json.dumps(payload),
        content_type="application/json"
    )

    # Assertions for the response
    assert response.status_code == 200
    predictions = response.json
    assert isinstance(predictions, list)
    assert len(predictions) == 2  # Two rows in the dataset
    assert all(isinstance(pred, int) for pred in predictions)


def test_prediction_endpoint_with_invalid_path(client):
    """
    Test the /prediction endpoint with an invalid test dataset path.
    """
    # Prepare the request payload with a non-existent path
    payload = {
        "test_data_path": "/nonexistent/path/to/testdata.csv"
    }

    # Send a POST request to the /prediction endpoint
    response = client.post(
        "/prediction",
        data=json.dumps(payload),
        content_type="application/json"
    )

    # Assertions for the response
    assert response.status_code == 400
    error_message = response.json
    assert "error" in error_message
    assert "does not exist" in error_message["error"]


def test_prediction_endpoint_with_missing_path(client):
    """
    Test the /prediction endpoint without providing a test_data_path.
    """
    # Send a POST request without a test_data_path
    response = client.post(
        "/prediction",
        data=json.dumps({}),
        content_type="application/json"
    )

    # Assertions for the response
    assert response.status_code == 400
    error_message = response.json
    assert "error" in error_message
    assert "No 'test_data_path' provided" in error_message["error"]

    
def test_scoring_endpoint(client):
    """
    Test the /scoring endpoint.
    """
    response = client.get("/scoring")
    assert response.status_code == 200
    assert "f1_score" in response.json
    assert isinstance(response.json["f1_score"], (int, float))


def test_summarystats_endpoint(client):
    """
    Test the /summarystats endpoint.
    """
    response = client.get("/summarystats")
    assert response.status_code == 200
    assert "summary_statistics" in response.json
    summary_stats = response.json["summary_statistics"]
    assert len(summary_stats) == 3  # Means, Medians, Stds
    assert all(isinstance(stat, list) for stat in summary_stats)


def test_diagnostics_endpoint(client):
    """
    Test the /diagnostics endpoint.
    """
    response = client.get("/diagnostics")
    assert response.status_code == 200
    diagnostics = response.json
    assert "execution_time" in diagnostics
    assert "missing_data_percentages" in diagnostics
    assert "outdated_packages" in diagnostics
    assert isinstance(diagnostics["execution_time"], dict)
    assert "data_ingestion_time" in diagnostics["execution_time"]
    assert "model_training_time" in diagnostics["execution_time"]
    assert isinstance(diagnostics["missing_data_percentages"], list)
    assert isinstance(diagnostics["outdated_packages"], list)

