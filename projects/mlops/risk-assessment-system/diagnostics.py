import pickle

import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

##################Function to get model predictions
def model_predictions(test_data, trained_model_file):
    #read the deployed model and a test dataset, calculate predictions
    
    # Prepare the test data
    X_test = test_data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]

    # Load the trained model
    with open(trained_model_file, 'rb') as f:
        model = pickle.load(f)

    # Make predictions using the test data
    y_pred = model.predict(X_test)

    return y_pred.tolist() #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary(dataset_path):    
    # Load the dataset
    dataset_path = os.path.join(dataset_path, 'finaldata.csv')
    data = pd.read_csv(dataset_path)

    # Select numeric columns
    numeric_columns = data.select_dtypes(include=[np.number])

    # Calculate summary statistics: mean, median, std for each numeric column
    means = numeric_columns.mean().tolist()
    medians = numeric_columns.median().tolist()
    stds = numeric_columns.std().tolist()

    # Combine the statistics into a list
    summary_statistics = [means, medians, stds]

    return summary_statistics

##################Function to get missing data information
def dataframe_missing_data(dataset_path):
    # Load the dataset
    dataset_path = os.path.join(dataset_path, 'finaldata.csv')
    data = pd.read_csv(dataset_path)

    # Calculate the percentage of missing values for each column
    missing_percentages = (data.isna().sum() / len(data) * 100).tolist()

    return missing_percentages

##################Function to get timings
def execution_time():
    # List to store timing values
    timings = []

    # Time the ingestion.py script
    start_time = timeit.default_timer()
    subprocess.run(['python', 'ingestion.py'], check=True)
    ingestion_time = timeit.default_timer() - start_time
    timings.append(ingestion_time)

    # Time the training.py script
    start_time = timeit.default_timer()
    subprocess.run(['python', 'training.py'], check=True)
    training_time = timeit.default_timer() - start_time
    timings.append(training_time)

    # Return the timings as a list
    return timings

##################Function to check dependencies
def outdated_packages_list():
    # Run the pip command to get a list of outdated packages
    result = subprocess.run(['pip', 'list', '--outdated'],
                            stdout=subprocess.PIPE, text=True, check=True)

    # Parse the output
    lines = result.stdout.strip().split('\n')
    packages = []

    # Skip the header lines (assume first two lines are header)
    for line in lines[2:]:  # Start from the third line
        parts = line.split()
        if len(parts) >= 3:
            package_name = parts[0]
            current_version = parts[1]
            latest_version = parts[2]
            packages.append([package_name, current_version, latest_version])

    # Convert to a DataFrame for tabular representation
    df = pd.DataFrame(packages, columns=['Package', 'Current Version', 'Latest Version'])

    # Return the DataFrame
    return df



if __name__ == '__main__':
    
    # read the dataset_csv_path as a pandas df and pass it to model_predictions
    # Load the test data
    test_data_file = os.path.join(config['test_data_path'], 'testdata.csv')  # Assuming test data file is named 'testdata.csv'
    test_data = pd.read_csv(test_data_file)

    trained_model_file = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    predictions = model_predictions(test_data, trained_model_file)
    print("\nModel Predictions:")
    print(predictions)
    
    summary = dataframe_summary(config['output_folder_path'])
    print("\nDataframe Summary (Means, Medians, Stds):")
    print(summary)
    
    timings = execution_time()
    print("\nExecution Times (Seconds):")
    print(f"Data Ingestion: {timings[0]} seconds, Model Training: {timings[1]} seconds")
    
    outdated_packages = outdated_packages_list()
    print("\nOutdated Packages:")
    print(outdated_packages)
