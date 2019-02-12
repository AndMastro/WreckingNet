# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:50:34 2019

@author: Ale & Mastro
"""

#FILEFORMT = '.png'
import tensorflow as tf

#classes = ['concrete_mixer/ConcreteMixer_onsite/', 'dozer_JD700J/JD700J_onsite', 'excavator_JD50G/JD50G_onsite', 'grader_JD670G/JD670G_onsite']

class rawCNN(tf.keras.Model):

    def __init__(self):
        super(rawCNN, self).__init__()

        self.convPre1 = tf.layers.Conv2D(filters=40,
                                      kernel_size=[1, 8],
                                      strides=(1, 1),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.convPre2 = tf.layers.Conv2D(filters=40,
                                      kernel_size=[1, 8],
                                      strides=(1, 1),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.poolPre1 = tf.layers.MaxPooling2D(pool_size=[1, 128], 
                                            strides=[1,128])
        self.conv1 = tf.layers.Conv2D(filters=24,
                                      kernel_size=[6, 6],
                                      strides=(1, 1),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(filters=24,
                                      kernel_size=[6, 6],
                                      strides=(1, 1),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.Conv2D(filters=48,
                                      kernel_size=[5, 5],
                                      strides=(2, 2),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv4 = tf.layers.Conv2D(filters=48,
                                      kernel_size=[5, 5],
                                      strides=(2, 2),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv5 = tf.layers.Conv2D(filters=64,
                                      kernel_size=[4, 4],
                                      strides=(2, 2),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)

        self.dense = tf.layers.Dense(200, activation=tf.nn.relu)
        self.dropout = tf.layers.Dropout(0.3)  # to be improved

        self.logits = tf.layers.Dense(units=5) #logits must suit the number of classes

    def call(self, x, training=False):
        #print(x.shape)
        x = self.convPre1(tf.reshape(x, [x.shape[0], 1, -1, 1]))
        #print(x.shape)
        x = self.poolPre1(self.convPre2(x))
        #print(x.shape)
        x = self.conv1(tf.reshape(x, [x.shape[0], 40, -1, 1]))
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.conv4(x)
        #print(x.shape)
        x = self.conv5(x)
        #print(x.shape)
        x = self.dense(tf.reshape(x, [x.shape[0], -1]))
        #print(x.shape)
        x = self.dropout(x, training=training)
        #print(x.shape)
        #print("=============")
        return self.logits(x)
