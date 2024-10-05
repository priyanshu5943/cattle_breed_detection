#Extract test data features.
import matplotlib.pyplot as plt
import seaborn as sns




import os
import gc

from sklearn.model_selection import train_test_split


import tensorflow as tf

import numpy as np #
import pandas as pd 

from keras import Sequential
from keras.callbacks import EarlyStopping

from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from keras.layers import Lambda, Input, GlobalAveragePooling2D,BatchNormalization
from keras.utils import to_categorical
# from keras import regularizers
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.applications.inception_v3 import InceptionV3, preprocess_input as InceptionV3_preprocess_input
from keras.applications.xception import Xception, preprocess_input as Xception_preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as InceptionResNetV2_preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input as NASNetLarge_preprocess_input


#function to extract features from the dataset by a given pretrained model
img_size = (331,331,3)

def get_features(model_name, model_preprocessor, input_size, data):

    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    
    #Extract feature.
    
    feature_maps = feature_extractor.predict(data, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps

def extact_features(data):
    inception_features = get_features(InceptionV3, InceptionV3_preprocess_input, img_size, data)
    xception_features = get_features(Xception, Xception_preprocess_input, img_size, data)
    nasnet_features = get_features(NASNetLarge, NASNetLarge_preprocess_input, img_size, data)
    inc_resnet_features = get_features(InceptionResNetV2, InceptionResNetV2_preprocess_input, img_size, data)

    final_features = np.concatenate([inception_features,
                                     xception_features,
                                     nasnet_features,
                                     inc_resnet_features],axis=-1)
    
   
    
    #deleting to free up ram memory
    del inception_features
    del xception_features
    del nasnet_features
    del inc_resnet_features
    gc.collect()
    
    
    return final_features

