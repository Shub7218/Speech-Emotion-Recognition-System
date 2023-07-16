import numpy as np
import librosa
import streamlit as st
import tensorflow as tf
import io
import sounddevice as sd
import soundfile as sf

# Load the model
# replace 'my_model.h5' with your own saved model
model = tf.keras.models.load_model('my_model.h5')

# Define emotions
emotions = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Define function to extract MFCC features
def extract_mfcc(audio_bytes, sr):
    y, sr = librosa.load(io.BytesIO(audio_bytes), duration=3, offset=0.5, sr=sr)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Define Streamlit app
st.set_page_config(page_title='Emotion Recognition', page_icon=':microphone:', layout='wide')
st.title('Emotion Recognition')
st.markdown('## Upload a .wav file or record your voice to recognize the emotion')
col1, col2 = st.columns(2)

# Get file from user
uploaded_file = col1.file_uploader('Upload .wav file', type='wav')

# If file is uploaded
if uploaded_file is not None:
    # Read the audio file as bytes
    audio_bytes = uploaded_file.read()

    # Extract MFCC features
    sr = None
    mfcc = extract_mfcc(audio_bytes, sr)
    mfcc = np.reshape(mfcc, newshape=(1, 40))

    # Make prediction using the model
    predictions = model.predict(mfcc)
    emotion = emotions[np.argmax(predictions[0])]

    # Display the predicted emotion
    col2.success('Predicted emotion: {}'.format(emotion))

# Record audio
recording = col1.button('Record')
if recording:
    duration = 4  # seconds
    sr = 44100
    myrecording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    with st.spinner('Recording...'):
        sd.wait()
    sf.write('temp.wav', myrecording, sr)
    audio_bytes = io.BytesIO()
    with open('temp.wav', 'rb') as f:
        audio_bytes.write(f.read())

    # Extract MFCC features
    mfcc = extract_mfcc(audio_bytes.getvalue(), sr)
    mfcc = np.reshape(mfcc, newshape=(1, 40))

    # Make prediction using the model
    predictions = model.predict(mfcc)
    emotion = emotions[np.argmax(predictions[0])]

    # Display the predicted emotion
    col2.success('Predicted emotion: {}'.format(emotion))
