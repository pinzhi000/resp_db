import streamlit as st
import pandas as pd
import numpy as np 
import sklearn
import os 

import io
from pathlib import Path
import librosa
import librosa.display

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

import audiomentations
from scipy.io import wavfile
import wave
import pydub

# deep learning 
import keras 
import tensorflow 
import tensorflow_text as text

# pickle! 
import pickle


# define global variables 
# set path 
path = os.path.dirname(__file__)


# define global functions 

# plot audio waveform func
def plot_wave(y, sr):
    fig, ax = plt.subplots(figsize=(14,5))
    
    # Visualize a waveform in the time domain
        # source: https://librosa.org/doc/main/generated/librosa.display.waveshow.html
    img = librosa.display.waveshow(y, sr=sr, x_axis="time")

    return plt.gcf()

# plot spectrogram func
def plot_spectrogram (Xdb, sr):
    fig, ax = plt.subplots(figsize=(20,7))
    
    # asdfasfd
        # source: asfafd
    img = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

    return plt.gcf()

# process audio file func 
    # ingest audio file and change it into numpy array to feed into model 

def audio_features(filename): 
    sound, sample_rate = librosa.load(filename)
    stft = np.abs(librosa.stft(sound))  
 
    mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40),axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate),axis=1)
    mel = np.mean(librosa.feature.melspectrogram(sound, sr=sample_rate),axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate),axis=1)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate),axis=1)
    
    concat = np.concatenate((mfccs,chroma,mel,contrast,tonnetz))
    return concat




# create 5 pages in streamlit app 
    # page 1: introduction

# add market research tab 
app_mode = st.sidebar.selectbox('Select Page', ['Introduction', 'Patient Dashboard', 'Modeling Accuracy Dashboard', 'Real-time Prediction', 'Hardware Build'])

if app_mode == 'Introduction':
    st.title("Project Background")
    st.markdown("Dataset :")



if app_mode == 'Real-time Prediction':
    st.title("Patient Lung Diagnostics")

    # drag and drop file uploader 
    uploaded_file = st.file_uploader("Choose an Audio File", type=[".wav", ".wave", ".flac", ".mp3", ".ogg"], accept_multiple_files=False)

    # play breathing audio 
    if uploaded_file is not None:

        st.text("")
        st.text("")
        st.caption("#### Play Uploaded Patient Audio")

        # audio player 
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav') # https://discuss.streamlit.io/t/how-to-save-file-uploaded-mp3-and-wav-files-using-streamlit/6920/15  

        # input audio file is .wav
        if uploaded_file.name.endswith('wav'):
    
            file_type = 'wav'
            # https://audiosegment.readthedocs.io/en/latest/audiosegment.html
            audio = pydub.AudioSegment.from_wav(uploaded_file)

            # export user uploaded file to app folder
            audio.export(path+'/'+uploaded_file.name, format=file_type)
 

        # load an audio file as a floating point time series
            # source: https://librosa.org/doc/main/generated/librosa.load.html
            # y: audio time series (np.ndarray); y=sound 
            # sr: sampling rate of y (scalar)
        y, sr = librosa.load(uploaded_file.name)
        
        # display waveplot of uploaded audio file
        st.pyplot(plot_wave(y, sr))

        st.text("")
        st.caption("#### Audio Spectrogram Visual")

        # display spectrogram
        # STFT (Short-time Fourier transform) represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows
        X = librosa.stft(y)
        Xdb = librosa.amplitude_to_db(abs(X))
        st.pyplot(plot_spectrogram(Xdb, sr))

        # to do: add sample files people can upload on the sidebar 

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


        # front-end code for prediction 
        st.markdown("---")
        st.caption("## Classify Patient Respiratory Audio")

        if st.button("Predict"):
            
            audio_array = audio_features(uploaded_file.name)
            st.write(audio_array)
            
            # make prediction
            picklefile = open("./rsd-model.pkl", "rb") 
            model = pickle.load(picklefile)

            prediction = model.predict(audio_array)
            st.write(prediction)

        # spinning 

        # collapsable -- explain model used and overall accuracy of model used for prediction 









