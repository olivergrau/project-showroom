from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    """Test the GET method on the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Salary Prediction API!"}

def test_predict_less_than_50k():
    """Test POST method with input that predicts <=50K."""
    input_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}

def test_predict_more_than_50k():
    """Test POST method with input that predicts >50K."""
    input_data = {
        "age": 56,
        "workclass": "Local-gov",
        "fnlgt": 216851,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
