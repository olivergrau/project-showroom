import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 


#################Function for training the model
def train_model(output_folder_path, output_model_path):
    # Load the dataset
    final_data_path = os.path.join(output_folder_path, 'finaldata.csv')
    data = pd.read_csv(final_data_path)

    # Prepare the data
    X = data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y = data["exited"]

    # Split the data into training and testing sets (optional, for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialize the logistic regression model
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='ovr', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Ensure the output folder for the model exists, create it if not
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
        print(f"Created model output folder at {output_model_path}.")

    # Save the trained model to the specified directory
    trained_model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    with open(trained_model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model training complete. Trained model saved to {trained_model_path}.")

if __name__ == '__main__':
    train_model(config['output_folder_path'], config['output_model_path'])
