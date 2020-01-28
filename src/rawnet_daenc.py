# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:50:34 2019

@author: Ale, Mastro & Umberto
"""

import tensorflow as tf
import random
import json
import sys
import datetime
import numpy as np

import matplotlib.pyplot as plt

from PickleGenerator import get_samples_and_labels
from utils import get_class_numbers, get_reduced_set, load, plot_confusion_matrix

def create_model(input_shape = (664,), encoding_size = 200):
    net_input = tf.keras.Input(shape = input_shape)
    x = tf.keras.layers.Reshape((1,664,1))(net_input)
    x = tf.keras.layers.Conv2D(filters=40,
                                kernel_size=[1, 8],
                                strides=(1, 1),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="E_ConvPre1")(x)
    x = tf.keras.layers.Conv2D(filters=40,
                                kernel_size=[1, 8],
                                strides=(1, 1),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="E_ConvPre2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=[1, 128],
                                        strides=[1, 128],
                                        name="E_MaxPoolPre1")(x)
    x = tf.keras.layers.Reshape((40,5,1))(x)
    x = tf.keras.layers.Conv2D(filters=24,
                                kernel_size=[6, 6],
                                strides=(1, 1),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="E_Conv1")(x)
    x = tf.keras.layers.Conv2D(filters=24,
                                kernel_size=[6, 6],
                                strides=(1, 1),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="E_Conv2")(x)
    x = tf.keras.layers.Conv2D(filters=48,
                                kernel_size=[5, 5],
                                strides=(2, 2),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="E_Conv3")(x)
    x = tf.keras.layers.Conv2D(filters=48,
                                kernel_size=[5, 5],
                                strides=(2, 2),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="E_Conv4")(x)
    x = tf.keras.layers.Conv2D(filters=64,
                                kernel_size=[4, 4],
                                strides=(2, 2),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="E_Conv5")(x)
    x = tf.keras.layers.Flatten()(x)
    encoding = tf.keras.layers.Dense(encoding_size,
                                        activation=tf.nn.relu,
                                        name="X_DenseOut1")(x)
    ####
    # decoding net - to be done
    x = tf.keras.layers.Reshape((5,1,40))(encoding)
    x = tf.keras.layers.Conv2DTranspose(filters=48,
                                kernel_size=[4, 4],
                                strides=(2, 2),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="D_Conv5")(x)
    x = tf.keras.layers.Conv2DTranspose(filters=48,
                                kernel_size=[5, 5],
                                strides=(2, 2),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="D_Conv4")(x)
    x = tf.keras.layers.Conv2DTranspose(filters=24,
                                kernel_size=[5, 5],
                                strides=(2, 2),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="D_Conv3")(x)
    x = tf.keras.layers.Conv2DTranspose(filters=24,
                                kernel_size=[6, 6],
                                strides=(1, 1),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="D_Conv2")(x)
    x = tf.keras.layers.Conv2DTranspose(filters=1,
                                kernel_size=[6, 6],
                                strides=(1, 1),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="D_Conv1")(x)
    x = tf.keras.layers.Reshape((1,8,40))(x)
    x = tf.keras.layers.UpSampling2D((1, 83),
                                        name="D_Upsample")(x) # 1, 662, 40
    x = tf.keras.layers.Conv2DTranspose(filters=40,
                                kernel_size=[1, 8],
                                strides=(1, 1),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="D_ConvPost2")(x) # 1, 662, 40
    x = tf.keras.layers.Conv2DTranspose(filters=1,
                                kernel_size=[1, 8],
                                strides=(1, 1),
                                kernel_initializer=tf.initializers.glorot_uniform,
                                padding="same",
                                activation=tf.nn.relu,
                                name="D_ConvPost1")(x) # 1, 662, 1
    #x = tf.keras.layers.ZeroPadding2D(((0,0),(-1,-2)))(x)
    net_output = tf.keras.layers.Flatten()(x) # 662
    
    # autoencode model
    model = tf.keras.Model(inputs=net_input, outputs=net_output, name="Autoencoder-RawNet")
    
    #encode model
    encoder = tf.keras.Model(inputs=net_input, outputs=encoding, name="RawNet-Encoder")
    
    #decoder model
    encoded_input = tf.keras.layers.Input((encoding_size,))
    deco = model.layers[-11](encoded_input)
    deco = model.layers[-10](deco)
    deco = model.layers[-9](deco)
    deco = model.layers[-8](deco)
    deco = model.layers[-7](deco)
    deco = model.layers[-6](deco)
    deco = model.layers[-5](deco)
    deco = model.layers[-4](deco)
    deco = model.layers[-3](deco)
    deco = model.layers[-2](deco)
    deco = model.layers[-1](deco)
    decoder = tf.keras.Model(inputs=encoded_input, outputs=deco, name="RawNet-Decoder")

    return model, encoder, decoder

if __name__ == "__main__":

    # Callback for tensorboard
    logdir = "..\\logs\\caenc\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

    # create autoencoder model
    model_net, encoder, decoder = create_model()
    #model_net.summary()
    #encoder.summary()
    #decoder.summary()
    
    model_net.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['mae','mse',])

    # Load dataset
    dataset_dict_path = "../dataset/pickles/ms30_hop15/classes.json"
    dataset_pickle_path_train = "../dataset/pickles/ms30_hop15/train.p"
    dataset_pickle_path_test = "../dataset/pickles/ms30_hop15/test.p"

    with open(dataset_dict_path, 'r') as fin:
        classes = json.load(fin)

    print(classes)

    train_set = load(dataset_pickle_path_train)
    test_set = load(dataset_pickle_path_test)

    # Normalize dataset length
    train_lens = get_class_numbers(train_set, classes)
    train_data = get_reduced_set(train_set, train_lens, 'min')
    test_lens = get_class_numbers(test_set, classes)
    test_data = get_reduced_set(test_set, test_lens, 'min')

    Xtrain, _, _ = get_samples_and_labels(train_data)
    Xtest, _, _ = get_samples_and_labels(test_data)

    

    print("Train size", len(Xtrain))
    print("Test size", len(Xtest))

    # add 0 before and after
    print("Padding Dataset")
    
    Xtrain = [np.append(x, [0, 0]) for x in Xtrain]
    Xtrain = np.array(Xtrain)
    Xtest = [np.append(x, [0, 0]) for x in Xtest]
    Xtest = np.array(Xtest)
    Ytrain = Xtrain
    Ytest = Xtest

    print("Convesion Done")


    model_net.fit(Xtrain,Ytrain, 64, epochs=10, callbacks=[tensorboard_callback])

    #Missing savemodel