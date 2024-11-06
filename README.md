# Automatic Speech Recognition

## 1. Using pretrained 🤗 facebook/wav2vec2-large-960h model
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

!!!!!!!!!!!!!
Place common_voice in main directory.
assume cloned to home directory at ~