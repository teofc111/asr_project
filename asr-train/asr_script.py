# %% [markdown]
# ## Tuning facebook:wav2vec2-large-960h

# %% [markdown]
# Here, we finetune the facebook:wav2vec2-large-960h model from huggingface using the `cv-valid-train` common_voice dataset. This notebook follows the finetuning framework from this [hugginface blog](https://huggingface.co/blog/fine-tune-wav2vec2-english) with minor adaptations. First, we import the required libraries.

# %%
# Imports
import os
import re
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import gc
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import Audio as PlayAudio

from accelerate import Accelerator
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, Audio, DatasetDict, load_from_disk, Dataset
import evaluate

import torch
from torch.utils.data import DataLoader
import torchaudio
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from pydub import AudioSegment
import soundfile as sf
from mutagen import File

from jiwer import wer

HOME_DIR = os.path.expanduser('~')

# %% [markdown]
# ### Pre-processing

# %% [markdown]
# We first convert all mp3 files to wav files, which the wav2vec2 model assumes. Additionally converting transcript texts to upper case to match the original model. This may take some time.

# %%
# File locations. All files assumed placed in asr_proejct folder
audio_or_dir = os.path.join(HOME_DIR,'asr_project/common_voice/cv-valid-train/')
audio_dir = os.path.join(HOME_DIR,'asr_project/common_voice/cv-valid-train/cv-valid-train/')
audioloc_transcript_or_dir = os.path.join(HOME_DIR,'asr_project/common_voice/cv-valid-train.csv')
audioloc_transcript_dir = os.path.join(HOME_DIR,'asr_project/asr-train/temp.csv')

# %%
# Function to convert mp3 to wav
def convert_mp3_to_wav(mp3_file):
    # Generate the output wav file path
    wav_file = mp3_file.replace('.mp3', '.wav')
    
    # Convert mp3 to wav if wav file does not exist
    if not os.path.exists(wav_file):
        waveform, sample_rate = torchaudio.load(mp3_file)
        torchaudio.save(wav_file, waveform, sample_rate)
    
    return wav_file


df = pd.read_csv(audioloc_transcript_or_dir)

# Convert mp3 to wav. Change mp3 file extension in df accordingly
df['filename'] = df['filename'].apply(
    lambda filename: convert_mp3_to_wav(
        os.path.join(audio_or_dir, filename)))

# Put texts to uppercase to match pre-finetuned model
df['text'] = df['text'].str.upper()
df['filename'] = df['filename'].map(lambda x: os.path.basename(x))

df_transcript = df

# %% [markdown]
# Checking audio file characteristics ...

# %%
def get_audio_info(file_path):
    # Extract filename and extension
    file_name, file_ext = os.path.splitext(os.path.basename(file_path))
    file_size = os.path.getsize(file_path)  # Size in bytes

    # Try to get audio length with mutagen
    try:
        audio = File(file_path)
        audio_length = audio.info.length if audio and audio.info else None
    except Exception as e:
        print(f"Could not process file {file_name}: {e}")
        audio_length = None

    return {
        'filename': file_name,
        'extension': file_ext,
        'size_bytes': file_size,
        'length_seconds': audio_length
    }

def process_directory(directory):
    # List all audio files in directory
    audio_files = [
        os.path.join(directory, f) for f in os.listdir(directory) 
        if os.path.isfile(os.path.join(directory, f))
    ]

    # Use tqdm with multiprocessing
    with Pool(cpu_count()) as pool:
        # Wrap audio files list with tqdm for progress bar
        audio_info = list(tqdm(pool.imap(get_audio_info, audio_files), total=len(audio_files), desc="Processing files"))

    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(audio_info)
    return df

# Get audio file information
audio_df = process_directory(audio_dir)
audio_df_mp3 = audio_df.loc[audio_df['extension']=='.mp3'].copy()
audio_df_wav = audio_df.loc[audio_df['extension']=='.wav'].copy().drop(columns=['length_seconds'])
audio_df_wav = audio_df_wav.merge(audio_df_mp3[['filename','length_seconds']], on='filename', how='left')
audio_df_wav['filename'] = audio_df_wav['filename'].map(lambda x: x+'.wav')
audio_df_wav.head()

# %% [markdown]
# We see that some of them have very high durations, up to 6 minutes long.

# %%
audio_df_wav.describe()

# %% [markdown]
# Checking the transcript, we find that the longest line read is only 33 words long, which should not take that long to read.

# %%
df_transcript['len'] = df_transcript['text'].str.len()
df_transcript = df_transcript[['filename','len','text']]

filename_longest = df_transcript.loc[df_transcript['len']==df_transcript['len'].max(), 'filename'].item()
text_longest = df_transcript.loc[df_transcript['len']==df_transcript['len'].max(), 'text'].item()

print(f'Filename with longest transcript: {filename_longest}')
print(f'Longest transcript text: {text_longest}')

longest_clip_duration = audio_df_wav.loc[audio_df_wav['filename']==filename_longest,'length_seconds'].item()
print(f'Longest transcript duration: {longest_clip_duration}s')

# %% [markdown]
# The clip with the longest transcript is 11s long. Considering differences in reading speeds, we assume the longest legitimate script reading to be 15s long. __We discard all samples with durations above 15s__. This will help prevent memory issues during model finetuning. We drop a total of 397 samples, keeping ~195k samples, saving a copy as csv file for later reference.

# %%
df_transcript = df_transcript.merge(audio_df_wav[['filename', 'length_seconds']], on='filename', how='left')
(df_transcript['length_seconds'] > 15).sum().item(),  (df_transcript['length_seconds'] < 15).sum().item()

# %%
df_transcript = df_transcript.loc[df_transcript['length_seconds']<15].drop(columns=['len','length_seconds'])
df_transcript.to_csv(audioloc_transcript_dir, index=False)

# %% [markdown]
# We create a `DatasetDict` for easy access to train-val splits.

# %%
# Load csv file with wav filenames, complete path and create dataset
df = pd.read_csv(audioloc_transcript_dir)
df['filename'] = df['filename'].map(lambda x: os.path.join(audio_dir,x))
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("filename",
                              Audio(sampling_rate=16000))         # Cast audio files with 16kHz sampling rate

# train-val 70-30 split
dataset = dataset.train_test_split(test_size=0.3, seed=42)        # Split to train-val

# Final, combined dataset
dataset = DatasetDict({
    'train': dataset['train'],
    'val': dataset['test']})

dataset

# %% [markdown]
# We will make use of the tokenizer and processor from `facebook/wav2vec2-large-960h` in the model finetuning below. First, the transcripts are converted to the format expected by the model. The transcript have already been converted into uppercase earlier for this purpose. We insert start, end, and delimited tokens below.

# %%
# Following the style of facebook/wav2ec2-large-960h model
start_token = "<s>"
end_token = "</s>"
word_delimiter_token = "|"

# Define the preprocessing function
def preprocess_transcript(example):
    transcript = example['text']  # Assuming the column with text is named 'text'
    
    # Step 1: Replace multiple spaces with a single space
    transcript = re.sub(r'\s+', ' ', transcript)  # Remove extra spaces
    
    # Step 2: Add start and end tokens, and replace spaces with '|'
    processed_transcript = start_token + transcript.replace(" ", f"{word_delimiter_token}") + end_token
    
    return {"processed_text": processed_transcript}  # Return the processed text in a dictionary

# Apply the preprocessing to both train and validation splits
dataset = dataset.map(preprocess_transcript, remove_columns=["text"],num_proc=4)

# %%
dataset = dataset.rename_column("filename", "input_values")
dataset = dataset.rename_column("processed_text", "labels")

# %% [markdown]
# Then, we tokenize the transcripts and use the `input_values` and `labels` column names in the datasets.

# %%
# Load processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

def prepare_dataset(batch):
    # Process 'input_values' column for 1D waveform values
    batch["input_values"] = processor(batch["input_values"]["array"],
                                      sampling_rate=16000).input_values[0]
    
    # Process the 'labels' column to create 'labels' (text data)
    batch["labels"] = processor(text=batch["labels"]).input_ids
    
    return batch

# Map the dataset transformation to both 'train' and 'val' splits
dataset = dataset.map(prepare_dataset, num_proc=2)


# %%
# Save the dataset to a directory
dataset.save_to_disk("temp_dataset")
