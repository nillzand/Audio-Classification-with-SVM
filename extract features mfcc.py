import os
import wave
import csv
import librosa

# Columns of the CSV file
fields = ['File Name'] + ['MFCC_'+str(i+1) for i in range(20)]

# List of all files in the current directory
files = os.listdir()

# Find all files with the .wav extension
wav_files = [f for f in files if f.endswith('.wav')]

# Create a new CSV file
with open('audio_files.csv', 'w', newline='') as csv_file:
    # Create a writer object for writing to the CSV file
    writer = csv.writer(csv_file)
    # Write the column headers
    writer.writerow(fields)

    # Read each audio file using wave.open() and extract features
    for wav_file in wav_files:
        with wave.open(wav_file, 'rb') as wav:
            # Extract desired features using librosa
            y, sr = librosa.load(wav_file)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

            # Add the file name and desired features to the CSV file
            writer.writerow([wav_file] + list(mfccs.flatten()))
            # Process of reading the features of audio files goes here
            print(f'Features extracted for file {wav_file} successfully.')