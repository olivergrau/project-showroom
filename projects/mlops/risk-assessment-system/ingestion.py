import pandas as pd
import os
import json


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path):
    # Ensure the output folder exists, create it if not
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output folder at {output_folder_path}.")
        
    # Get list of all .csv files in the input folder path
    csv_files = [
        file for file in os.listdir(input_folder_path)
        if file.endswith('.csv')
    ]

    # List to store each DataFrame
    dataframes = []

    # Iterate through csv files, read them, and append to the dataframes list
    for csv_file in csv_files:
        file_path = os.path.join(input_folder_path, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate all DataFrames and remove duplicates
    combined_df = pd.concat(dataframes, ignore_index=True).drop_duplicates()

    # Save the final DataFrame to output path
    final_file_path = os.path.join(output_folder_path, 'finaldata.csv')
    combined_df.to_csv(final_file_path, index=False)

    # Save the list of ingested file names to ingestedfiles.txt as a Python list literal
    ingested_files_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(ingested_files_path, 'w') as f:
        f.write(str(csv_files))  # Write the Python list as a string literal

    print(f"Data ingestion complete. Final dataset saved to {final_file_path}.")
    print(f"Ingested file names saved to {ingested_files_path}.")


if __name__ == '__main__':
    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']

    merge_multiple_dataframe(input_folder_path, output_folder_path)
