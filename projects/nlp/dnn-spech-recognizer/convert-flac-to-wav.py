import os
import subprocess

# Function to convert a flac file to wav
def convert_flac_to_wav(flac_file):
    print("called convert_flac_to_wav")
    wav_file = flac_file.replace('.flac', '.wav')
    # Run the ffmpeg command to convert the file
    subprocess.run(['ffmpeg', '-i', flac_file, wav_file], check=True)
    print(f"Converted: {flac_file} -> {wav_file}")

# Function to traverse directories recursively and find all .flac files
def traverse_and_convert(root_dir):
    print(f"traverse_and_convert started: {root_dir}")
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.flac'):
                flac_file = os.path.join(dirpath, filename)
                convert_flac_to_wav(flac_file)

# Define the root directory where the dataset is stored
root_directory = "./data/nlpnd_projects/LibriSpeech"

# Start the conversion process
traverse_and_convert(root_directory)