import tensorflow
import keras.layers as kL
import numpy as np
from numpy import array
from numpy import argmax
import pandas as pd
import cv2
import os
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder

base_folder = os.getcwd()
height, width, channels = 256, 256, 3

def create_train_dataset(df, img_folder):
    IMAGE_ARRAY = []
    IMAGE_LABEL = []
    for rows in df[["ID","Label"]].values:
        ID, Label = rows[0], rows[1]
        print("Loading train images: ",ID)
        image_path = img_folder+ID
        img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        img = img.astype('float32')
        img /= 255
        IMAGE_ARRAY.append(img)
        IMAGE_LABEL.append(Label)
    print("create dataset complete.")
    return IMAGE_ARRAY, IMAGE_LABEL


def create_test_dataset(df, img_folder):
    IMAGE_ARRAY = []
    for rows in df[["ID"]].values:
        ID = rows[0]
        print("Loading test images: ",ID)
        image_path = img_folder+ID
        img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        img = img.astype('float32')
        img /= 255
        IMAGE_ARRAY.append(img)
    print("create dataset complete.")
    return IMAGE_ARRAY

def split_train_valid(data, label, data_num):
    train_data_num = int(data_num * 0.8)
    X_train, y_train, X_valid, y_valid = data[:train_data_num], label[:train_data_num], data[train_data_num:], label[train_data_num:]    
    return  X_train, y_train, X_valid, y_valid


def onehot_encode(inputs):
    onehot_encoder = OneHotEncoder(sparse = False)
    onehot_encoded = []
    for input_data in inputs:
        input_data = array(input_data)
        input_data = input_data.reshape(len(input_data), 1)
        onehot_encoded.append(onehot_encoder.fit_transform(input_data))
    return onehot_encoded, onehot_encoder

def onehot_decode(input_data, onehot_encoder):
    onehot_decoded = onehot_encoder.inverse_transform(input_data).reshape(len(input_data))
    return onehot_decoded


def create_model():
    model = Sequential([kL.Conv2D(filters = 32, kernel_size = 7, strides = 2, padding = "same", activation = "relu", input_shape = [height, width, channels]),
                        # 128
                        kL.MaxPooling2D(2),
                        # 64
                        kL.Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = "relu"),
                        kL.Conv2D(filters = 64, kernel_size = 3, padding = "same", activation = "relu"),
                        kL.MaxPooling2D(2), 
                        # 32
                        kL.Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = "relu"),
                        kL.Conv2D(filters = 128, kernel_size = 3, padding = "same", activation = "relu"),
                        kL.MaxPooling2D(2), 
                        # 16
                        kL.Flatten(),
                        kL.Dense(64, activation = "relu"),
                        kL.Dense(32, activation = "relu"),
                        kL.Dense(6,activation = "softmax")
                        ])
    model.compile(loss = "sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
    
    return model


train_df = pd.read_csv(os.getcwd()+"/train.csv")
test_df = pd.read_csv(os.getcwd()+"/test.csv")
image_array, image_label = create_train_dataset(train_df, base_folder+"/train_images/")
X_train, y_train_data, X_valid, y_valid_data = split_train_valid(image_array, image_label, len(image_array))
X_test = create_test_dataset(test_df, base_folder+"/test_images/")

onehot_output, onehot_encoder = onehot_encode([y_train_data, y_valid_data])
y_train, y_valid = onehot_output[0], onehot_output[1]

print(X_test[0])

'''
model = create_model()
print(model.summary())
'''

