import requests

# Base URL of your deployed service
BASE_URL = "https://showcase-census-cicd.onrender.com"

# Example payload for prediction
payload = {
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

# Function to test the root endpoint
def test_root():
    response = requests.get(BASE_URL + "/")
    if response.status_code == 200:
        print("Root endpoint response:", response.json())
    else:
        print(f"Root endpoint failed with status code {response.status_code}")

# Function to test the prediction endpoint
def test_prediction():
    response = requests.post(BASE_URL + "/predict", json=payload)
    if response.status_code == 200:
        print("Prediction response:", response.json())
    else:
        print(f"Prediction endpoint failed with status code {response.status_code}")
        print("Response content:", response.text)

if __name__ == "__main__":
    # Test root endpoint
    test_root()

    # Test prediction endpoint
    test_prediction()
