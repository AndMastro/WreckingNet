from __future__ import absolute_import, division, print_function

import sys
import pathlib
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import random
#import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import Model
from tensorflow.keras import regularizers

PICKLE_PATH = "..\\data\\genosAndSBPOneHot.p"
SAVE_PATH = "F:\\Repositories\\BioNet\\models\\autoEncoder\\"

dataset = {}
EPOCHS = 100
LEARNING_RATE = 0.001 #0.0005
BATCH_SIZE = 16

'''class regressionNet(tf.keras.Model):

    def __init__(self):
        super(regressionNet, self).__init__()

        self.dense1 = tf.layers.Dense(64, activation=tf.nn.relu)
        self.dense2 = tf.layers.Dense(64, activation=tf.nn.relu)

        #self.dropout = tf.layers.Dropout(0.3) 
        self.dense3 = tf.layers.Dense(1)  

    def call(self, x, training=False):
        x = self.dense1(tf.reshape(x, [x.shape[0], -1]))
        x = self.dense2(x)
        return self.dense3(x)'''

def build_model():

    genoInput = layers.Input(shape=(1, 54, 8), name = "genoInput")  # adapt this since using `channels_first` data format
    #x = layers.ZeroPadding2D(padding=((0, 13), (0, 0)), data_format="channels_first")(genoInput)
    x = layers.Conv2D(filters = 16, kernel_size = [2,2], activation = 'relu', padding='same', data_format = "channels_first", activity_regularizer=regularizers.l1(10e-5), input_shape=(1, 54, 8))(genoInput)
    x = layers.MaxPooling2D(pool_size = [3,2], padding='same', data_format = "channels_first")(x)
    x = layers.Conv2D(filters = 8, kernel_size = [2, 2], activation='relu', padding='same', data_format = "channels_first", activity_regularizer=regularizers.l1(10e-5))(x)
    x = layers.MaxPooling2D(pool_size = [3, 2], padding='same', data_format = "channels_first")(x)
    x = layers.Conv2D(filters = 8, kernel_size = [2, 2], activation='relu', padding='same', data_format = "channels_first", activity_regularizer=regularizers.l1(10e-5))(x)
    encoded = layers.MaxPooling2D(pool_size = [3, 2], padding='same', data_format = "channels_first")(x)

    # at this point the representation is (8, 6, 1) 

    x = layers.Conv2D(filters =8, kernel_size=(2, 2), activation='relu', padding='same', data_format = "channels_first")(encoded)
    x = layers.UpSampling2D((3, 2), data_format = "channels_first")(x)
    x = layers.Conv2D(filters = 8, kernel_size=(2, 2), activation='relu', padding='same', data_format = "channels_first")(x)
    x = layers.UpSampling2D((3, 2), data_format = "channels_first")(x)
    x = layers.Conv2D(filters =16, kernel_size=(2, 2), activation='relu',  padding='same', data_format = "channels_first")(x)
    x = layers.UpSampling2D((3, 2) , data_format = "channels_first")(x)
    decoded = layers.Conv2D(filters =1, kernel_size=(2, 2), activation='sigmoid', padding='same', data_format = "channels_first")(x)

    autoencoder = Model(genoInput, decoded)

    #optimizer = tf.keras.optimizers.Adadelta(lr=LEARNING_RATE)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.summary()
    
    return autoencoder


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  #plt.ylim([0,0.125])
  plt.ylim([0,22])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  #plt.ylim([0,0.025])
  plt.ylim([0,400])
  plt.legend()

  plt.show()




if __name__ == "__main__":
    model = build_model()
    trainX = []
    trainY = []
    phenoX = []

    with open(PICKLE_PATH, 'rb') as handle:
        dataset = pickle.load(handle)

    for ind in dataset:
        xProxy = dataset[ind][0][:-4]
        #xList = [x / 6400 for x in dataset[ind][0]]
        xList = xProxy
        #xList = dataset[ind][0] + [dataset[ind][1]]
        #add 4 rows of zero padding for autoencoder compatibility
        for i in range(0,3):
            xList = xList + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        trainX.append(xList)
        trainY.append(dataset[ind][1])
        phenoData = dataset[ind][0][-4:]
        phenoData[1] = (phenoData[1] - 57.1488) / 8.25298
        phenoData[2] = phenoData[1]**2
        phenoData[3] = (phenoData[3] - 27.3664) / 4.78212
        phenoX.append(phenoData) #normalize data
        

    featureAndLabels = list(zip(trainX, trainY, phenoX))
    random.shuffle(featureAndLabels)
    trainX, trainY, phenoX = zip(*featureAndLabels)
    
    #print(trainY)

    X_train =  np.array(trainX)
    #print(X_train.shape[0], X_train.shape[1])
    X_train = np.reshape(X_train, (X_train.shape[0], -1, 54, 8))
    #print(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    
    print(X_train.shape)
    print(X_train)
    #for i in range(0, X_train.shape[0]):
     # print(X_train[i])
     # X_train[i] = np.reshape(X_train[i], (51, -1), "C")
    #prova = X_train[0].reshape(51,8)
    #print(prova)

    #print(X_train.shape[0])
    #print(X_train.shape[1])

    #print(X_train[0].shape[0])
    #print(X_train[0].shape[1])
    
    Y_train = np.array(trainY)
    #print(Y_train.shape[0])
    #Y_train = np.reshape(Y_train, (Y_train.shape[0], -1))
    #print(Y_train.shape[0], Y_train.shape[1])
    X_pheno =  np.array(phenoX)
    X_pheno = np.reshape(X_pheno, (X_pheno.shape[0], 4))
    print(X_pheno)

    print(X_train.shape[0])
    print(Y_train.shape[0])
    print(X_pheno.shape[0])

    #sys.exit()

    saveModelCallback = callbacks.ModelCheckpoint(SAVE_PATH + "{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=10)

    history = model.fit([X_train], [X_train], #NORMALIZE DATA. IT SHOULD BE DONE
        epochs=EPOCHS, validation_split = 0.3, verbose=1 , batch_size=BATCH_SIZE)

    #model.save_weights(SAVE_PATH)

    plot_history(history)