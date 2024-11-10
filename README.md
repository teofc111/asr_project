# Automatic Speech Recognition
 
This repository contains a project for Automatic Speech Recognition (ASR), making use of `facebook:wav2vec2-large-960h`. We
1. Create a microservice for using the aforementioned model
2. Finetune the model on the common voice dataset
3. Provide a discussion on further steps to improve the model
4. Provide a discussion on a deep learning pipeline for a continuously learning ASR model meant for transcribing dysarthric speech
5. Perform hotword detection on the common voice dataset

## 1. Using pretrained 🤗 facebook/wav2vec2-large-960h model microservice
Please follow the steps below to install and run this model
```
cd asr
sudo docker build -t asr-api .
sudo docker run -p 8001:8001 asr-api
```
Once the Docker container is running, you can run the following in a new terminal to verify that the connection. You should receive 'pong' as a response.
`curl http://localhost:8001/ping`

To transcribe a .wav or .mp3 audio file, run the following command, replacing `/path/to/file.mp3` with the path to your file.
`curl -F ‘file=@/path/to/file.mp3’ http://localhost:8001/asr`

A sample of a batch-transcribed text is given in `asr/cv-valid-dev-updated.csv`.

Assumptions:
Common Voice dataset downloaded from common_voice in main directory.
assume cloned to home directory at ~
