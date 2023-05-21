filename = '/home/appuser/venv/lib/python3.9/site-packages/keras_vggface/models.py'
text = open(filename).read()
open(filename, 'w+').write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import tensorflow as tf
import pickle
import sys
import os
import pandas as pd 
from keras_vggface.utils import preprocess_input
from keras_vggface import VGGFace
from keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
import warnings
warnings.filterwarnings("ignore")
import av

# Load the VGGFace model
resnet = VGGFace(model = 'resnet50', include_top = False, input_shape = (224, 224, 3), pooling = 'avg')


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
font = cv2.FONT_HERSHEY_SIMPLEX


def preprocessing(img):
    
    sample = img.copy()
    sample = cv2.resize(sample, (224, 224))
    sample = np.array(sample).astype(np.float64)
    sample = np.expand_dims(sample, axis = 0)
    sample = preprocess_input(sample, version = 2)
    
    return sample

# Load the fine-tuned model weights
with open('best_model.pkl', 'rb') as f:
    new_model = pickle.load(f)


class VideoProcessor:

    def recv(self, frame):
        frm = frame.to_ndarray(format = 'bgr24')

        faces = faceCascade.detectMultiScale(
                cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY),
                scaleFactor = 1.15,
                minNeighbors = 5,
                minSize = (30,30),
            )
        for (x, y, w, h) in faces:
                # Draw a blue rectangle around the face
                cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 255, 0), 3)
                processed_img = preprocessing(frm[y:y+h, x:x+w])
                features = resnet.predict(processed_img)
                preds = new_model.predict(features)
                cv2.putText(frm, f'BMI: {preds}', (x+5, y-5), font, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(frm, format = 'bgr24') 

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://bpb-us-w2.wpmucdn.com/voices.uchicago.edu/dist/e/2560/files/2019/04/UChicago_Phoenix-Wallpaper-Gray.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
st.set_page_config(
    page_title = "BMI Prediction with Webcam",
    layout = "wide",
    initial_sidebar_state="expanded")

add_bg_from_url() 

st.write("# Use Webcam to Find Your BMI!🎚️")

webrtc_streamer(key = 'Use the Webcam', sendback_audio = False, video_processor_factory = VideoProcessor)


