import ast
import json
import os
import subprocess

import pandas as pd
from diagnostics import model_predictions, dataframe_summary, execution_time, dataframe_missing_data, outdated_packages_list
from scoring import score_model
from deployment import store_model_into_pickle
from training import train_model
from ingestion import merge_multiple_dataframe

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

def check_new_files(sourcedata_path, ingested_files):
    """
    Compare files in the sourcedata folder with the ingested file names.
    
    Args:
    - sourcedata_path (str): Path to the sourcedata folder.
    - ingestedfiles (list of str): List of already ingested file names.
    
    Returns:
    - dict: A dictionary with keys:
        - "new_files" (list): Files in sourcedata that are not in ingestedfiles.
        - "missing_files" (list): Files in ingestedfiles that are not in sourcedata.
    """
    # Get the list of files in the sourcedata folder
    sourcedata_files = os.listdir(sourcedata_path)

    # Find new files (in sourcedata but not in ingested files)
    new_files = [file for file in sourcedata_files if file not in ingested_files]

    # Find missing files (in ingested files but not in sourcedata)
    missing_files = [file for file in ingested_files if file not in sourcedata_files]

    return {
        "new_files": new_files,
        "missing_files": missing_files
    }

##################Check and read new data
#first, read ingestedfiles.txt
ingested_files = ast.literal_eval(
    open(
        os.path.join(config['prod_deployment_path'], "ingestedfiles.txt"), "r").read())

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
result = check_new_files(config['input_folder_path'], ingested_files)
print("New files:", result["new_files"])
print("Missing files:", result["missing_files"])


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if result["new_files"] == []:
    print("No new data found. Ending the process.")
    exit()

print("New data found. Proceeding.")
merge_multiple_dataframe(config['input_folder_path'], config['output_folder_path']) # ingest new data

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
latest_score = ast.literal_eval(
    open(
        os.path.join(config['prod_deployment_path'], "latestscore.txt"), "r").read())

print(f"Latest deployed model score: {latest_score}")

# get predictions with current deployed model and the newly ingested data and score it
df = pd.read_csv(os.path.join(config['output_folder_path'], "finaldata.csv"))
new_f1 = score_model(
    df, os.path.join(config['prod_deployment_path'], "trainedmodel.pkl"))

print(f"New F1 score: {new_f1}")

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_f1 <= latest_score:
    print(f"New F1 score {new_f1} is less or equal than the latest deployed model score {latest_score}. Proceeding.")
    exit()

###################Re-training
train_model(config['output_folder_path'], config['output_model_path'])

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
store_model_into_pickle(
    config['output_model_path'], config['prod_deployment_path'])

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
subprocess.run(['python', 'apicalls.py'], check=True)
subprocess.run(['python', 'reporting.py'], check=True)