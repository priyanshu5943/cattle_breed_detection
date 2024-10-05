import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import helper
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import streamlit as st

# Import specific models and preprocessing functions to avoid conflicts
from tensorflow.keras.applications import InceptionV3, Xception, InceptionResNetV2, NASNetLarge
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inceptionv3
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_nasnet

# Load the model
model = load_model('model.h5')
  # Update with actual weights file

# Streamlit app
st.title(" 🐮🐄 Cattle Breed Classifier")

# Upload image
uploaded_file = st.file_uploader("Choose a   🐄🐄 cattle image...")

if uploaded_file is not None:
    
    # Display image
    img_size = (331,331,3)
    img_g = load_img(uploaded_file,target_size = img_size)
    img_g = np.expand_dims(img_g, axis=0)
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    # Preprocess the image
    test_features = helper.extact_features(img_g)
    predg = model.predict(test_features)

    classes = [
    'Alambadi',
    'Amritmahal',
    'Ayrshire',
    'Banni',
    'Bargur',
    'Bhadawari',
    'Brown Swiss',
    'Dangi',
    'Deoni',
    'Gir',
    'Guernsey',
    'Hallikar'
]


    
    breed = classes[np.argmax(predg[0])]
    st.markdown(f"<h2><b>  🐮Predicted Cattle Breed: {breed} 🐄 </b></h2>", unsafe_allow_html=True)

    
