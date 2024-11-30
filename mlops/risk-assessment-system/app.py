from flask import Flask, jsonify, request
import pandas as pd
import json
import os
from diagnostics import model_predictions, dataframe_summary, execution_time, dataframe_missing_data, outdated_packages_list
import ast

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

####################### Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    Accepts a JSON payload with a file path to a test dataset (on the server),
    loads the dataset, calls model_predictions, and returns predictions.
    """
    # Parse incoming JSON data
    try:
        input_data = request.get_json()
        test_data_path = input_data.get("test_data_path")
        if not test_data_path:
            return jsonify({"error": "No 'test_data_path' provided in the request."}), 400

        # Load the test dataset from the given path
        if not os.path.exists(test_data_path):
            return jsonify({"error": f"The file path '{test_data_path}' does not exist."}), 400

        test_data = pd.read_csv(test_data_path)
    except Exception as e:
        return jsonify({"error": "Failed to load test dataset.", "details": str(e)}), 400

    # Call the model_predictions function and return results
    try:
        predictions = model_predictions(test_data, os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl'))
        return jsonify(predictions), 200
    except Exception as e:
        return jsonify({"error": "Failed to make predictions.", "details": str(e)}), 500


####################### Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    """
    Returns the F1 score of the deployed model.
    """
    # Get the F1 score from the deployed model
    f1_score = ast.literal_eval(open(os.path.join(config['prod_deployment_path'], 'latestscore.txt')).read())
    return jsonify({"f1_score": f1_score}), 200

####################### Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats():
    """
    Returns summary statistics (means, medians, stds) for the dataset.
    """
    # Call the dataframe_summary function and return the result
    summary_stats = dataframe_summary(config['output_folder_path'])
    return jsonify({"summary_statistics": summary_stats}), 200

####################### Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    """
    Returns diagnostics information: execution times, percent NA values, and outdated packages.
    """
    # Timing information
    timing_info = execution_time()

    # Missing data percentages
    missing_data_info = dataframe_missing_data(config['output_folder_path'])

    # Outdated packages
    outdated_packages = outdated_packages_list()

    return jsonify({
        "execution_time": {
            "data_ingestion_time": timing_info[0],
            "model_training_time": timing_info[1],
        },
        "missing_data_percentages": missing_data_info,
        "outdated_packages": outdated_packages.to_dict(orient="records")
    }), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
