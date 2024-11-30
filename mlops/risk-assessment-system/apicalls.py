import requests
import json
import os

with open('config.json', 'r') as f:
    config = json.load(f)

output_path = os.path.join(config['output_model_path'])

# Specify the base URL for the API
URL = "http://127.0.0.1:8000"

payload = {
    "test_data_path": "testdata/testdata.csv"
}

# Call each API endpoint and store the responses
response1 = requests.post(
    f"{URL}/prediction",
    json=payload
)
response2 = requests.get(f"{URL}/scoring")
response3 = requests.get(f"{URL}/summarystats")
response4 = requests.get(f"{URL}/diagnostics")

# Combine all API responses into a dictionary
responses = {
    "prediction": response1.json() if response1.status_code == 200 else {"error": response1.text},
    "scoring": response2.json() if response2.status_code == 200 else {"error": response2.text},
    "summarystats": response3.json() if response3.status_code == 200 else {"error": response3.text},
    "diagnostics": response4.json() if response4.status_code == 200 else {"error": response4.text},
}

# Write the responses to a file in the workspace
output_file = "apireturns.json"
with open(os.path.join(output_path, output_file), "w") as f:
    json.dump(responses, f, indent=4)

print(f"API responses written to {output_file}")




