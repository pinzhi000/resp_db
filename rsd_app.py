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
# import pickle

# progress bar
import time 

# test
#     # source: https://www.tensorflow.org/api_docs/python/tf/saved_model/LoadOptions
# tensorflow.saved_model.LoadOptions(experimental_io_device = '/job:localhost')


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
app_mode = st.sidebar.selectbox('Select Page', ['Introduction', 'Real-time Prediction', 'Patient Dashboard'])

if app_mode == 'Introduction':
    st.title("Project Background")
    st.markdown("Dataset :")

if app_mode == 'Patient Dashboard':
    st.title("Patient Dashboard")

    data = pd.read_csv("data/total_merge_ahmed.csv", index_col='pid', encoding="utf-8").drop("Unnamed: 0", axis=1)
    data = data.fillna(np.nan)

    pids = np.unique(data.index)

    patient = st.selectbox("Patient ID", pids)
    # st.write(data)
    # st.write(data.loc[[patient]])
    filtered_data = data.loc[[patient]]

    chest_location = st.selectbox("Chest Location", [*filtered_data['Chest location']])

    st.markdown('#')
    st.markdown('#')

    filtered_data = filtered_data[filtered_data['Chest location'] == chest_location]    

    
    st.markdown("### Audio Report & Diagnosis")
    st.markdown("""---""")
    if np.isnan(filtered_data.iloc[0, 2]):
        col1, col2, col3, col4 = st.columns(4)

        col3.metric("Child Weight (kg)", filtered_data.iloc[0, 3])
        col4.metric("Child Height (cm)", filtered_data.iloc[0, 4])
    else:
        col1, col2, col3 = st.columns(3)

        col3.metric("BMI (kg/m2)", filtered_data.iloc[0, 2])

    col1.metric("Age", int(filtered_data.iloc[0, 0]))
    col2.metric("Sex", filtered_data.iloc[0, 1])

    def sentence_case(sentenses): 
        words=sentenses.split(". ") 
        new=". ".join([word.capitalize() for word in words]) 
        return new

    re = sentence_case(filtered_data.iloc[0, -1])
    am = sentence_case(filtered_data.iloc[0, -2])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###### Acquisition Mode")
        am
    
    with col2:
        st.markdown("###### Recording Equipment")
        re

    st.markdown('#')
    st.markdown('#')
    
    
    st.markdown("### Patient Respiratory Audio")
    st.markdown("""---""")

    uploaded_file = open(f"audio_and_txt_files/{filtered_data.iloc[0, 6].strip()}.wav", 'rb')

    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes,
                format='audio/wav')  # https://discuss.streamlit.io/t/how-to-save-file-uploaded-mp3-and-wav-files-using-streamlit/6920/15

    # input audio file is .wav
    if uploaded_file.name.endswith('wav'):
        file_type = 'wav'
        # https://audiosegment.readthedocs.io/en/latest/audiosegment.html
        audio = pydub.AudioSegment.from_wav(uploaded_file)

        # export user uploaded file to app folder
        audio.export(path + '/' + uploaded_file.name, format=file_type)

    # load an audio file as a floating point time series
    # source: https://librosa.org/doc/main/generated/librosa.load.html
    # y: audio time series (np.ndarray); y=sound
    # sr: sampling rate of y (scalar)
    y, sr = librosa.load(uploaded_file.name)

    # display waveplot of uploaded audio file
    st.pyplot(plot_wave(y, sr))

    st.text("")
    st.markdown("### Audio Spectrogram Image")
    st.markdown("""---""")

    # display spectrogram
    # STFT (Short-time Fourier transform) represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    st.pyplot(plot_spectrogram(Xdb, sr))

    st.markdown("### Patient Pulmonary Diagnosis")
    st.markdown("""---""")

    st.text_area('', f'''Diagnosis: {filtered_data.iloc[0, 5]} \n - Placeholder for physician notes''')

if app_mode == 'Real-time Prediction':
    st.title("Patient Lung Diagnostics")

    def action(uploaded_file, selected_provided_file):
        sample_files = {"Sample 1":"105_1b1_Tc_sc_Meditron", "Sample 2":"226_1b1_Al_sc_Meditron", "Sample 3":"124_1b1_Al_sc_Litt3200"}

        if uploaded_file is not None:
            return uploaded_file

        if selected_provided_file in sample_files:
            return open(f"audio_and_txt_files/{sample_files[selected_provided_file]}.wav", 'rb')
        
        return None

    "Upload an audio file or choose a sample file from below"
    # drag and drop file uploader 
    uploaded_file = st.file_uploader("", type=[".wav", ".wave", ".flac", ".mp3", ".ogg"], accept_multiple_files=False)

    
    selected_provided_file = st.selectbox(
        label="", options=["Sample 1", "Sample 2", "Sample 3"],
        index=0
    )
    
    if st.button('Submit'):
        audio_file = action(uploaded_file, selected_provided_file)
        # play breathing audio 
        if audio_file is not None:

            st.text("")
            st.text("")
            st.caption("#### Play Uploaded Patient Audio")

            # audio player 
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav') # https://discuss.streamlit.io/t/how-to-save-file-uploaded-mp3-and-wav-files-using-streamlit/6920/15  

            # input audio file is .wav
            if audio_file.name.endswith('wav'):
        
                file_type = 'wav'
                # https://audiosegment.readthedocs.io/en/latest/audiosegment.html
                audio = pydub.AudioSegment.from_wav(audio_file)

                # export user uploaded file to app folder
                audio.export(path+'/'+audio_file.name, format=file_type)
    

            # load an audio file as a floating point time series
                # source: https://librosa.org/doc/main/generated/librosa.load.html
                # y: audio time series (np.ndarray); y=sound 
                # sr: sampling rate of y (scalar)
            y, sr = librosa.load(audio_file.name)
            
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

            # loading spinner 
            with st.spinner('Calculating...'):
                time.sleep(4)

            audio_array = audio_features(audio_file.name)
            audio_array = audio_array.reshape(193, 1)

            # transform audio array to feed into deep learning model 
            audio_file_reshaped = np.reshape(audio_array, [1, 193, 1, 1])


            # note: save tf model as .h5 file directly to avoid cloud deployment issues 
                # source: https://www.tensorflow.org/guide/keras/save_and_serialize#:~:text=The%20recommended%20format%20is%20SavedModel,'h5'%20to%20save()%20.
                # source: https://discuss.streamlit.io/t/oserror-savedmodel-file-does-not-exist-at/12985
            model = tensorflow.keras.models.load_model('my_model_test.h5') #set filepath for Github 

    #             # loading spinner 
    #             with st.spinner('Calculating...'):
    #                 time.sleep(3)

            prediction_num = np.argmax(model.predict(audio_file_reshaped))

            # convert numbers to diseases 
                # {"COPD":0, "Healthy":1, "URTI":2, "Bronchiectasis":3, "Pneumonia":4, "Bronchiolitis":5, "Asthma":6, "LRTI":7}
            if prediction_num == 0:
                class_pred = "Chronic Obstructive Pulmonary Disease (COPD)"
            elif prediction_num == 1:
                class_pred = "Healthy"
            elif prediction_num == 2:
                class_pred = "Upper Respiratory Tract Infection (URTI)"
            elif prediction_num == 3:
                class_pred = "Bronchiectasis"
            elif prediction_num == 4:
                class_pred = "Pneumonia"
            elif prediction_num == 5:
                class_pred = "Bronchiolitis"
            elif prediction_num == 6:
                class_pred = "Asthma"
            elif prediction_num == 7:
                class_pred = "Lower Respiratory Tract Infection (LRTI)"
            
            # write classification to front-end 
            st.markdown(f"**Patient Diagnosis:** {class_pred}")

        # expander section to provide details around prediction 
        with st.expander("See details"):
            st.write("**Step 1:** ingest uploaded patient audio file")
            st.write("**Step 2:** extract specific features from audio file and place them in an array structure (shown below)")
            st.write(audio_array)
            st.write("**Step 3:** push above array through Convolutional Neural Network (CNN) deep learning model to obtain prediction (CNN model structure shown below)")
            st.image('./CNN/Model Structure Layers.png')
            st.markdown(f"**Step 4:** Above model predicts / diagnoses the uploaded patient audio file with **{class_pred}**")
            st.write("**Step 5:** CNN model has overall predictive training accuracy of ~86%")
            st.image('./CNN/acc_loss.png')
            
            
                











