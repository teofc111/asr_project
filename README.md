# Automatic Speech Recognition
 
This repository contains a project for Automatic Speech Recognition (ASR), making use of `facebook:wav2vec2-large-960h`.
1. Create a microservice for using the aforementioned model
2. Finetune the model on the common voice dataset
3. Provide a discussion on further steps to improve the model
4. Perform hotword detection on the common voice dataset.
5. Provide a discussion on a Self-Supervised Learning (SSL) pipeline for a continuously learning ASR model for transcribing dysarthric speech

Please ensure that:
1. The repository is cloned to your home directory, i.e. `~/`.
2. The Common Voice dataset has been downloaded from [here](https://www.dropbox.com/scl/fi/i9yvfqpf7p8uye5o8k1sj/common_voice.zip?rlkey=lz3dtjuhekc3xw4jnoeoqy5yu&dl=0) and placed in the main repository, i.e. `~/asr_project/common_voice`. This dataset is not provided directly here in view of its large file size.
3. Please use Python 3.12.7 if possible and install the dependencies by running the shell command below.
```
pip install -r requirements.txt
```
4. Please install ffmpeg by running the shell command below.
```
sudo apt-get install ffmpeg
```


## 1. Using pretrained 🤗 facebook/wav2vec2-large-960h model microservice
Please follow the steps below to install and run the `facebook/wav2vec2-large-960h` model (__task 2a__), which has been containerized (__task 2e__).
```
cd asr
sudo docker build -t asr-api .
sudo docker run -p 8001:8001 asr-api
```
- Once the Docker container is running, you can run the following in a new terminal to verify that the connection. You should receive 'pong' as a response (__task 2b__).
`curl http://localhost:8001/ping`
- To transcribe a .wav or .mp3 audio file, run the following command, replacing `/path/to/file.mp3` with the path to your file.
`curl -F ‘file=@/path/to/file.mp3’ http://localhost:8001/asr`. A sample of a batch-transcribed text is given in `asr/cv-valid-dev-updated.csv` (__task 2d__).
- The API is given by `asr_api.py`  (__task 2c__).

## 2. Finetuning 🤗 facebook/wav2vec2-large-960h model
Files related to this task is placed in `asr_project/asr-train`.
- The model finetuning process (__task 3a__) is given in `asr_project/asr-train/cv_train_2a.ipynb`. The finetuned model achieved a Word Error Rate (WER) of 7.1% against the `cv-valid-test` split of the common voice dataset, as shown in the notebook (__task 3c__).
- The final finetuned model is given in `asr_project/asr-train/wav2vec2-large-960h-cv` (__task 3b__).

## 3. Further improving finetuned model
The performance of the finetuned model is compared to the performance of its pre-finetuned counterpart, yielding a WER score of 7.6% (former) vs 10.8% (latter). A discussion on steps for further improvement is provided in `asr_project/training-report.pdf` (__task 4__).

## 4. Hotword detection on common voice dataset
Files related to this task is placed in `asr_project/hotword-detection`.
Hotword detection is performed directly via keyword match and through text embedding comparisons using the `hkunlp/instructor-large` model from [here](https://huggingface.co/hkunlp/instructor-large).
- Keyword match is performed in `asr_project/hotword-detection/cv-hotword-5a.ipynb` and the detected mp3 files recorded in `asr_project/hotword-detection/detected.txt` (__task 5a__).
- Keyword similarity checks using text embeddings is done in `asr_project/hotword-detection/cv-hotword-similarity-5b.ipynb` and the output placed in `asr_project/hotword-detection/cv-valid-dev-updated.csv` (__task 5a__)

## 5. Self-supervised learning pipeline for model to transcribe dysarthric speech
The pipeline for an SSL pipeline for creating a model to transcribe dysarthric speech is provided in `asr_project/essay-ssl.pdf` (__task 6__).