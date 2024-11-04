'''
Microservice for Automatic Speech Recognition (ASR).
Using huggingface facebook/wav2vec2-large-960h model.
'''


import os
from flask import Flask, request, jsonify

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# from datasets import load_dataset
import torch
import torchaudio
from pydub import AudioSegment

# Flask app
app = Flask('asr')

# Load the model and processor when needed
model = None
processor = None

# PING endpoint
@app.route('/ping')
def ping():
    return 'pong'

# ASR inference endpoint
@app.route('/asr', methods=['POST'])
def asr():
    global model, processor

    # Check if a file is part of the request
    if ('file' not in request.files) or (request.files['file'].filename == ''):
        return jsonify({"error": "Please upload an audio file (.mp3/.wav)."})
    
    file = request.files['file']

    # Handle different file types
    temp_wav_path = "temp.wav"                           # To save temporarily as wav file
    if file.filename.endswith('.mp3'):
        temp_mp3_path = 'temp.mp3'
        file.save(temp_mp3_path)
        audio = AudioSegment.from_mp3(temp_mp3_path)
        audio.export(temp_wav_path, format="wav")
        os.remove(temp_mp3_path)                         # Delete temp mp3 file
    elif file.filename.endswith('.wav'):
        file.save(temp_wav_path)
    else:
        return jsonify({"error": "Please upload only .mp3/.wav audio files."})

    # Load model and processor
    if model is None or processor is None:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

    # Load audio file and resample (if needed)
    waveform, sample_rate = torchaudio.load(temp_wav_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Get duration (s)
    duration = waveform.size(1) / 16000

    # Process audio and infer
    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", padding="longest", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    # Delete temp wav file
    os.remove(temp_wav_path)                             # Delete temp wav file

    return jsonify({"transcription": transcription, "duration":duration})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001)


