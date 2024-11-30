import pandas as pd
import json
import os
from diagnostics import model_predictions
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    test_data_file = os.path.join(config['test_data_path'], 'testdata.csv')  # Assuming test data file is named 'testdata.csv'
    test_data = pd.read_csv(test_data_file)
    
    predictions = model_predictions(test_data, os.path.join(config['output_model_path'], 'trainedmodel.pkl'))

    # Extract the ground truth (target values)
    ground_truth = test_data["exited"]

    # Calculate the confusion matrix
    cm = confusion_matrix(ground_truth, predictions)

    # Plot and save the confusion matrix as a PNG
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues, colorbar=True)

    # Save the plot to the output_model_path
    output_path = os.path.join(config['output_model_path'], 'confusion_matrix.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix saved as {output_path}")


if __name__ == '__main__':
    score_model()
