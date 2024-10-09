import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm.autonotebook import tqdm

import numpy as np #
import pandas as pd 

from keras import Sequential
from keras.callbacks import EarlyStopping
import helper
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from keras.layers import Lambda, Input, GlobalAveragePooling2D,BatchNormalization
from keras.utils import to_categorical
# from keras import regularizers
from keras.models import Model
from keras.preprocessing.image import load_img
from PIL import Image
from keras.models import load_model
import streamlit as st
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input

# Load the model
model = load_model('C:\Users\Nasem\Pictures\cattle_breed_detection_app\cattle_breed_model.h5')
  # Update with actual weights file

# Streamlit app
st.title("Cattle Breed Classifier")

# Upload image
uploaded_file = st.file_uploader("Choose a cattle image...")

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
    st.write(f"Predicted Cattle Breed: {breed}")




    
