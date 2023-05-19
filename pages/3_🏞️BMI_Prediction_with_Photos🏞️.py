import streamlit as st
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
from PIL import Image
warnings.filterwarnings("ignore")

# Load the VGGFace model
vggface = VGGFace(model = 'vgg16', include_top = True, input_shape = (224, 224, 3), pooling = 'avg')

# retrieve fc6 layer
fc6 = vggface.get_layer('fc6')
new_vgg16 = Model(inputs = vggface.input, outputs = fc6.output)

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

def prediction(img):
    image = load_img(img)
    image = np.array(image)

    faces = faceCascade.detectMultiScale(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            scaleFactor = 1.15,
            minNeighbors = 5,
            minSize = (30,30),
        )
    for (x, y, w, h) in faces:
            # Draw a blue rectangle around the face
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            processed_img = preprocessing(image[y:y+h, x:x+w])
            features = new_vgg16.predict(processed_img)
            preds = new_model.predict(features)
            cv2.putText(image, f'BMI: {preds}', (x+5, y-5), font, 3, (255, 255, 255), 4)

    return Image.fromarray(image)
    

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
    page_title = "BMI Prediction",
    layout = "wide",
    initial_sidebar_state="expanded")

add_bg_from_url() 

st.write("# Upload an Image to Find Your BMI!🎚️")

uploaded_file = st.file_uploader(' ', type = ['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    # process the file
    processed_img = prediction(uploaded_file)

    # display the prediction
    st.image(processed_img)