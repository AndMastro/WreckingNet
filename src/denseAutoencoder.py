import tensorflow as tf

import json

import random

import matplotlib.pyplot as plt

from PickleGenerator import get_samples_and_labels
from utils import get_class_numbers, get_reduced_set, load, plot_confusion_matrix

tf.enable_eager_execution()

class DenseAutoencoder(tf.keras.Model):

    def __init__(self):
        super(DenseAutoencoder, self).__init__()

        self.enc1 = tf.layers.Dense(662)
        self.enc2 = tf.layers.Dense(500)
        self.enc3 = tf.layers.Dense(330)
        self.enc4 = tf.layers.Dense(165)
        self.enc5 = tf.layers.Dense(5)
        self.dec1 = tf.layers.Dense(5)
        self.dec2 = tf.layers.Dense(165)
        self.dec3 = tf.layers.Dense(330)
        self.dec4 = tf.layers.Dense(500)
        self.dec5 = tf.layers.Dense(662)

    def call(self, input):
        x = self.enc1(input)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        return x

    def encode(self, input):
        x = self.enc1(input)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        return x

    def decode(self, input):
        x = self.dec1(input)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        return x

if __name__ == "__main__":
    dataset_dict_path = "../dataset/pickles/ms30_hop15/classes.json"
    dataset_pickle_path_train = "../dataset/pickles/ms30_hop15/train.p"
    dataset_pickle_path_test = "../dataset/pickles/ms30_hop15/test.p"

    with open(dataset_dict_path, 'r') as fin:
        classes = json.load(fin)
    print(classes)

    train_set = load(dataset_pickle_path_train)
    test_set = load(dataset_pickle_path_test)

    random.shuffle(train_set)
    random.shuffle(test_set)  # should be here

    train_lens = get_class_numbers(train_set, classes)
    train_data = get_reduced_set(train_set, train_lens, 'min')

    test_lens = get_class_numbers(test_set, classes)
    test_data = get_reduced_set(test_set, test_lens, 'min')

    Xtrain, _, Ytrain = get_samples_and_labels(train_data)
    Xtest, _, Ytest = get_samples_and_labels(test_data)

    print("Train size", len(Ytrain))
    print("Test size", len(Ytest))

    print("Build Dataset")

    Xtrain = tf.convert_to_tensor(Xtrain, dtype=tf.float32)
    Ytrain = tf.convert_to_tensor(Ytrain, dtype=tf.float32)

    Xtest = tf.convert_to_tensor(Xtest, dtype=tf.float32)
    Ytest = tf.convert_to_tensor(Ytest, dtype=tf.float32)

    print("Convesion Done")

    train_it = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
    test_it = tf.data.Dataset.from_tensor_slices((Xtest, Ytest))

    def loss(net, x, y):
        return tf.keras.losses.binary_crossentropy(y_true = y, y_pred= net(x))

    opt = tf.train.AdamOptimizer()

    print("Let's try")
    net = DenseAutoencoder()

    trainAcc = []
    testAcc = []
    lossValues = []

    epochs = 10
    batch_size = 64

    print("Start")

    for epoch in range(epochs):

        for xb, _ in train_it.batch(batch_size):
            yb = xb
            ypred = net(xb)

            lossValue = tf.keras.losses.binary_crossentropy(y_pred=ypred, y_true=yb)
            lossValues = lossValues + list(lossValue)

        for xb, yb in train_it.shuffle(1000).batch(batch_size):
            opt.minimize(lambda: loss(net, xb, xb))

        print(epoch)

    plt.plot(lossValues)
    plt.show()