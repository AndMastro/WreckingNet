# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:50:34 2019

@author: Ale & Mastro
"""

import tensorflow as tf


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
                                               strides=[1, 4])

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

        self.logits = tf.layers.Dense(units=5)  # logits must suit the number of classes

    def call(self, x, training=False):
        #print("rawnet - input", x.shape)
        x = self.convPre1(tf.reshape(x, [x.shape[0], 1, -1, 1]))
        #print("rawnet - convpre1", x.shape)
        x = self.convPre2(x)
        #print("rawnet - convpre2", x.shape)
        x = self.poolPre1(x)
        #print("rawnet - poolpre1", x.shape)
        x = self.conv1(tf.reshape(x, [x.shape[0], 40, -1, 1]))
        #print("rawnet - conv1", x.shape)
        x = self.conv2(x)
        #print("rawnet - conv2", x.shape)
        x = self.conv3(x)
        #print("rawnet - conv3", x.shape)
        x = self.conv4(x)
        #print("rawnet - conv4", x.shape)
        x = self.conv5(x)
        #print("rawnet - conv5", x.shape)
        x = self.dense(tf.reshape(x, [x.shape[0], -1]))
        #print("rawnet - dense", x.shape)
        x = self.dropout(x, training=training)
        #print("rawnet - dropout", x.shape)
        return self.logits(x)
