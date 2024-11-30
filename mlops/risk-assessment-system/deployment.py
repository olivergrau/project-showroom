import shutil
import os
import json

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

####################function for deployment
def store_model_into_pickle(output_model_path, prod_deployment_path):
    # Ensure the deployment folder exists, create it if not
    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)
        print(f"Created production deployment folder at {prod_deployment_path}.")

    # Define the source paths for the required files
    model_file = os.path.join(output_model_path, 'trainedmodel.pkl')
    score_file = os.path.join(output_model_path, 'latestscore.txt')
    ingested_files = os.path.join(config['output_folder_path'], 'ingestedfiles.txt')

    # Define the destination paths in the deployment directory
    dest_model_file = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    dest_score_file = os.path.join(prod_deployment_path, 'latestscore.txt')
    dest_ingested_files = os.path.join(prod_deployment_path, 'ingestedfiles.txt')

    # Copy the files to the deployment directory
    shutil.copy2(model_file, dest_model_file)
    shutil.copy2(score_file, dest_score_file)
    shutil.copy2(ingested_files, dest_ingested_files)

    print(f"Files successfully deployed to {prod_deployment_path}:")
    print(f"- Model: {dest_model_file}")
    print(f"- Score: {dest_score_file}")
    print(f"- Ingested Files: {dest_ingested_files}")

if __name__ == '__main__':
    store_model_into_pickle(config['output_folder_path'], config['prod_deployment_path'])        
        
        

