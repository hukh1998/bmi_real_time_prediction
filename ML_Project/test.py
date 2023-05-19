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
vggface = VGGFace(model = 'vgg16', include_top = True, input_shape = (224, 224, 3), pooling = 'avg')

# retrieve fc6 layer
fc6 = vggface.get_layer('fc6')
new_vgg16 = Model(inputs = vggface.input, outputs = fc6.output)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
font = cv2.FONT_HERSHEY_SIMPLEX

def preprocessing(img):
    sample = img.to_ndarray()

    faces = faceCascade.detectMultiScale(
                cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY),
                scaleFactor = 1.15,
                minNeighbors = 5,
                minSize = (30,30),
            )
    
    for (x, y, w, h) in faces:
        # Draw a blue rectangle around the face
        cv2.rectangle(sample, (x, y), (x+w, y+h), (0, 255, 0), 3)
        processed_img = preprocessing(sample[y:y+h, x:x+w])
        features = new_vgg16.predict(processed_img)
        preds = new_model.predict(features)
        cv2.putText(sample, f'BMI: {preds}', (x+5, y-5), font, 1, (255, 255, 255), 2)

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
                features = new_vgg16.predict(processed_img)
                preds = new_model.predict(features)
                cv2.putText(frm, f'BMI: {preds}', (x+5, y-5), font, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(frm, format = 'bgr24') 
    
    
def main():

    st.title("Testing")

    upload = st.button('Upload an Image')
    webcam = st.button('Use the Webcam')

    if upload: 
        uploaded_file = st.file_uploader('Upload an Image', type = ['png', 'jpg'])
        if uploaded_file is not None:
            # display the prediction
            st.image(uploaded_file)


    if webcam:
        webrtc_streamer(key = 'key', sendback_audio = False, video_processor_factory = VideoProcessor)

    
if __name__ == '__main__':
    main()


