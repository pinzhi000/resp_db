# import streamlit as st
# import pandas as pd
# import numpy as np 
# import sklearn
# import os 

# import io
# from pathlib import Path
# import librosa
# import librosa.display

# import numpy as np
# import streamlit as st
# from matplotlib import pyplot as plt

# import audiomentations
# from scipy.io import wavfile
# import wave
# import pydub

# # deep learning 
# import keras 
# import tensorflow 
# import tensorflow_text as text

# # pickle! 
# # import pickle

# # progress bar
# import time 

# # test
# #     # source: https://www.tensorflow.org/api_docs/python/tf/saved_model/LoadOptions
# # tensorflow.saved_model.LoadOptions(experimental_io_device = '/job:localhost')


# # define global variables 
# # set path 
# path = os.path.dirname(__file__)


# # define global functions 

# # plot audio waveform func
# def plot_wave(y, sr):
#     fig, ax = plt.subplots(figsize=(14,5))
    
#     # Visualize a waveform in the time domain
#         # source: https://librosa.org/doc/main/generated/librosa.display.waveshow.html
#     img = librosa.display.waveshow(y, sr=sr, x_axis="time")

#     return plt.gcf()

# # plot spectrogram func
# def plot_spectrogram (Xdb, sr):
#     fig, ax = plt.subplots(figsize=(20,7))
    
#     # asdfasfd
#         # source: asfafd
#     img = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

#     return plt.gcf()

# # process audio file func 
#     # ingest audio file and change it into numpy array to feed into model 

# def audio_features(filename): 
#     sound, sample_rate = librosa.load(filename)
#     stft = np.abs(librosa.stft(sound))  
 
#     mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40),axis=1)
#     chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate),axis=1)
#     mel = np.mean(librosa.feature.melspectrogram(sound, sr=sample_rate),axis=1)
#     contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate),axis=1)
#     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate),axis=1)
    
#     concat = np.concatenate((mfccs,chroma,mel,contrast,tonnetz))
#     return concat




# # create 5 pages in streamlit app 
#     # page 1: introduction

# # add market research tab 
# app_mode = st.sidebar.selectbox('Select Page', ['Introduction', 'Real-time Prediction', 'Patient Dashboard'])

# if app_mode == 'Introduction':
#     st.title("Project Background")
#     st.markdown("Dataset :")


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


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{

        background-color: #d8f9ff
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

        # background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        # background-size: cover


# create 5 pages in streamlit app 
    # page 1: introduction

# add market research tab 
app_mode = st.sidebar.selectbox('Select Page', ['Introduction', 'Real-time Prediction', 'Patient Dashboard'])

if app_mode == 'Introduction':
    # add_bg_from_url() 
#     add_bg_from_local('./LSTM/1980.jpg')   


    # st.title("Project Background")



    st.text("")
    st.markdown("### Project Background")
    st.markdown("""

Digital stethoscopes are used to record sounds emitted from patients’ lungs including respiratory or breathing audio.
Physicians can leverage these digital audio files to diagnose patients with ailments such as chronic pulmonary obstructive disease (COPD). 
The problem our team is solving revolves around processing a large collection of audio files in an effort to accurately diagnose patients with certain lung diseases using deep learning models. 
We have embedded our algorithms into a MVP fullstack application. 

Our software tool has market potential since it can be deployed at hospitals, doctors' offices, and medical schools across the country. The model output classifies each patient’s audio with
one of the following diagnoses: **COPD, Healthy, URTI, Bronchiectasis, Pneumonia, or Bronchiolitis**. 
Our algorithm’s classifier can be utilized to affirm, contradict, or further investigate a patient's lung disease diagnosis.

""")
    st.image('./LSTM/1980.jpg')



    st.text("")
    st.markdown("### Dataset")
    st.markdown("""
The dataset used for this project was commissioned by the 2017 International Conference on Biomedical Health Informatics (ICBHI 2017). 
The provided database contains audio samples collected by 2 independent bioinformatics research teams over a time period of several years. 
Specifically, the database is comprised of 920 annotated respiratory audio files recorded from 126 test subjects.  In total, our team listened to 5.5 hours of patient breathing audio spanning 6898 respiratory cycles. 
""")


    
#     st.text("")
#     st.markdown("### Approach Methodology ")
#     st.markdown("""
#     I write articles about Data Science, Python and related topics. 
#     The articles are mostly written on the Medium platform.
    
#     You can find my articles [here](https://alan-jones.medium.com)
#     and if you would like to know when I publish new ones, you can 
#     sign up for an email alert on my Medium 
#     [page](https://alan-jones.medium.com/subscribe).
#     Below are a few articles you might find interesting...
# """)



    st.text("")


    st.markdown("### Feature Engineering ")
    st.markdown("""  Predictive features were extracted from each patient audio recording using a python library called Librosa. 
    Our AI engineers extracted the following 5 key features: mel-frequency cepstral coefficients, chromagram, mel-scaled spectrogram, spectral contrast, and tonal centroid features.
    We then stored the above results in numerical form via numpy arrays.  These arrays capture critical information such as respiratory oscillations, pitch content, amplitude of breathing noises, peaks and valleys in audio, and chord sequences from .wav audio files.""")
    st.image('./LSTM/Pic1.png')



    st.text("")
    st.markdown("### Methodology & Results")
    st.markdown("""

    Our team programmed 4 customized neural networks capable of ingesting patient audio recordings and spitting out accurate diagnostic predictions.  The algorithms are:
    - Convolutional Neural Network (CNN)
    - Long Short-term Memory Artificial Neural Network (LSTM)
    - CNN ensembled with Unidirectional LSTM
    - CNN ensembled with Bidirectional LSTM
    
    We experimented with the 4 neural networks’ layering structures, hyperparameters, checkpoint values, and early
    stopping parameters to produce optimal lung disease classification results. The algorithms were designed using python libraries tensorflow and keras. 

    Evaluation metrics of accuracy, precision, recall and F1-score were used to determine the best performing model.  The **Sequential LSTM neural network** with diagram architecture presented below provided the highest overall predictive accuracy at approximately 98%  
""")

    st.markdown(""" . """)

    st.image('./LSTM/Pic2.png')
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
        sample_files = {"Patient Audio Sample 1":"105_1b1_Tc_sc_Meditron", "Patient Audio Sample 2":"226_1b1_Al_sc_Meditron", "Patient Audio Sample 3":"124_1b1_Al_sc_Litt3200"}

        if uploaded_file is not None:
            return uploaded_file

        if selected_provided_file in sample_files:
            return open(f"audio_and_txt_files/{sample_files[selected_provided_file]}.wav", 'rb')
        
        return None

    "Upload patient audio file or choose sample file from below"
    # drag and drop file uploader 
    uploaded_file = st.file_uploader("", type=[".wav"], accept_multiple_files=False)

    selected_provided_file = st.selectbox(
        label="", options=["Patient Audio Sample 1", "Patient Audio Sample 2", "Patient Audio Sample 3"],
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
            st.write("**Step 3:** push above array through LSTM Neural Network deep learning model to obtain prediction (model structure shown below)")
            st.image('./LSTM/Model Structure Layers.png')
            st.markdown(f"**Step 4:** Above model predicts / diagnoses the uploaded patient audio file with **{class_pred}**")
            st.write("**Step 5:** LSTM model has overall predictive training accuracy of ~98%")
            st.image('./LSTM/acc_loss.png')
            
            
                











