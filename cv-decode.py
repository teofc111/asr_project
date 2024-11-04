'''
Decode all audio files in cv-valid-dev
Assumes flask app is already running
'''
import os
import requests
import glob
import pandas as pd

# Define paths/URLs
url = 'http://localhost:8001/asr'
cv_valid_dev_folder_path = '/home/tfc/asr/common_voice/cv-valid-dev/cv-valid-dev/*'
cv_valid_dev_csv_path = '/home/tfc/asr/common_voice/cv-valid-dev.csv'
cv_valid_dev_updated_csv_path = '/home/tfc/asr/cv-valid-dev-updated.csv'

# Create df from original csv to update with generated text
df = pd.read_csv(cv_valid_dev_csv_path)
df = df.set_index('filename',drop=True)
df['generated_text'] = ''

# Get all audio filepaths
audio_filepaths = glob.glob(cv_valid_dev_folder_path)
audio_filenames = ['cv-valid-dev/'+os.path.basename(file) for file in audio_filepaths]

# Make POST request to flask app
j=0
for audio_filepath, audio_filename in zip(audio_filepaths, audio_filenames):
    with open(audio_filepath, 'rb') as audio_file:
        response = requests.post(url, files={'file': audio_file})
        df.loc[audio_filename,'generated_text'] = response.json()['transcription']
        j+=1
        if j %100 == 0:
            print(f'{j}/{len(audio_filepath)}')

df.to_csv(cv_valid_dev_updated_csv_path) # Saving to current (asr) folder