import pandas as pd
import pickle
import os
from sklearn import metrics
import json


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
    

#################Function for model scoring
def score_model(test_data_df, trained_model_file):
    # Ensure the output folder exists, create it if not
    output_model_path = config['output_model_path']
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
        print(f"Created model output folder at {output_model_path}.")

    # Prepare the test data
    X_test = test_data_df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y_test = test_data_df["exited"]

    # Load the trained model
    with open(trained_model_file, 'rb') as f:
        model = pickle.load(f)

    # Make predictions using the test data
    y_pred = model.predict(X_test)

    # Calculate the F1 score
    f1_score = metrics.f1_score(y_test, y_pred)

    # Save the F1 score to latestscore.txt (to the output model path, NOT the production path!)
    score_file = os.path.join(output_model_path, 'latestscore.txt')
    with open(score_file, 'w') as f:
        f.write(str(f1_score))

    print(f"Model scoring complete. F1 score: {f1_score}")
    print(f"F1 score saved to {score_file}.")
    
    return f1_score

if __name__ == '__main__':
    test_data_file = os.path.join(config['test_data_path'], 'testdata.csv')  # Assuming test data file is named 'testdata.csv'    
    test_data = pd.read_csv(test_data_file)

    trained_model_file = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
    
    score_model(test_data, trained_model_file)
