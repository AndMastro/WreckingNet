import tensorflow as tf
import tensorflow.contrib.eager as tfe

import sys
import json
import random

import matplotlib.pyplot as plt
import numpy as np

from rawnet import rawCNN
from spectronet import SpectroCNN
from utils import get_class_numbers, get_reduced_set, load, plot_confusion_matrix
from PickleGenerator import get_samples_and_labels


tf.enable_eager_execution()

model_raw_path = "../models/30/raw.h5"
model_spectro_path = "../models/30/spectro.h5"

model_raw = rawCNN()
model_spectro = SpectroCNN()

# predict one batch
try:
    with open('config.json', mode='r', encoding='utf-8') as fin:
        params = json.load(fin)
except Exception as e:
    print(e)
    print("No config file, aborting...")
    sys.exit(0)

raw_path = params['RAW_MODEL_PATH']
spectro_path = params['SPECTRUM_MODEL_PATH']

train_dataset_path = params['TRAIN_PICKLE']
test_dataset_path = params['TEST_PICKLE']

batch_size = 16

try:
    with open(params['DICT_JSON'], mode='r', encoding='utf-8') as fin:
        class_dict = json.load(fin)
except Exception as e:
    print(e)
    print("We have lost correpsondances, aborting...")
    sys.exit(0)

train_set = load(train_dataset_path)
test_set = load(test_dataset_path)
if train_set is None:
    print("No Test data, aborting...")
    sys.exit(0)

random.shuffle(train_set)
train_lens = get_class_numbers(train_set, class_dict)
train_data = get_reduced_set(train_set, train_lens, 'min')

Xtrain_raw, Xtrain_spec, Ytrain = get_samples_and_labels(train_data)

Xtrain_raw = tf.convert_to_tensor(Xtrain_raw, dtype=tf.float32)
Xtrain_spec = tf.convert_to_tensor(Xtrain_spec, dtype=tf.float32)
Ytrain = tf.convert_to_tensor(Ytrain, dtype=tf.int32)

random.shuffle(test_set)
test_lens = get_class_numbers(test_set, class_dict)
test_data = get_reduced_set(test_set, test_lens, 'min')

Xtest_raw, Xtest_spec, Ytest = get_samples_and_labels(test_data)

size = len(Ytest)

Xtest_raw = tf.convert_to_tensor(Xtest_raw, dtype=tf.float32)
Xtest_spec = tf.convert_to_tensor(Xtest_spec, dtype=tf.float32)
Ytest = tf.convert_to_tensor(Ytest, dtype=tf.int32)

print("Allocating tensors")

train_it = tf.data.Dataset.from_tensor_slices((Xtrain_raw, Xtrain_spec, Ytrain))
test_it = tf.data.Dataset.from_tensor_slices((Xtest_raw, Xtest_spec, Ytest))

def _parse_example(x, y, z):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    z = tf.cast(z, tf.int32)
    return x, y, z

for x, z, yb in test_it.batch(1):
    break

_ = model_raw(x)
_ = model_spectro(z)

model_raw.load_weights(model_raw_path)
model_spectro.load_weights(model_spectro_path)

class combinedCNN(tf.keras.Model):

    def __init__(self, spectro_model, raw_model):
        super(combinedCNN, self).__init__()

        self.raw = raw_model
        self.spectro = spectro_model

        self.extra = tf.layers.Dense(units = 5, name="ExtraLayer", trainable = True)

    def call(self, x, y):
        raw_out = self.raw(x)
        spectro_out = self.spectro(y)
        #print(raw_out.shape, spectro_out.shape)
        out = tf.concat([raw_out, spectro_out], axis = 1)
        out = self.extra(out)
        return out

model = combinedCNN(model_spectro, model_raw)

for xb, x2b, yb in train_it.batch(1):
    _ = model(xb, x2b)
    break

def loss(net, x, z, y):
    return tf.losses.sparse_softmax_cross_entropy(logits=net(x, z), labels=y)

opt = tf.train.AdamOptimizer(learning_rate=0.005)

print(tf.trainable_variables())

trainAcc = []
testAcc = []
lossValues = []

for epoch in range(10):

    accTrain = tfe.metrics.SparseAccuracy()
    for xb, x2b, yb in train_it.batch(batch_size):
        ypred = model(xb, x2b)
        accTrain(predictions=ypred, labels=yb)
        lossValue = tf.losses.sparse_softmax_cross_entropy(logits=ypred, labels=yb)
        lossValues.append(lossValue)

    accTest = tfe.metrics.SparseAccuracy()
    for xb, x2b, yb in test_it.batch(batch_size):
        ypred = model(xb,x2b)
        accTest(predictions=ypred, labels=yb)

    trainAcc.append(accTrain.result().numpy())
    testAcc.append(accTest.result().numpy())
    print('==================================')
    print('Train accuracy at epoch {} is {} %'.format(epoch, trainAcc[-1] * 100))
    print('Test accuracy at epoch {} is {} %'.format(epoch, testAcc[-1] * 100))
    print('Loss value at epoch {} is {}'.format(epoch, lossValue))
    print('==================================')

    for xb, x2b, yb in train_it.shuffle(1000).batch(batch_size):
        opt.minimize(lambda: loss(model, xb, x2b, yb), var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ExtraLayer"))

    plt.plot(trainAcc)
    plt.title('Spectrum Train Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
    plt.plot(testAcc)
    plt.title('Spectrum Test Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
    plt.plot(lossValues)
    plt.title('Spectrum  Loss Value')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()

    pred = []
    true = []

    for xb, x2b, yb in test_it.batch(batch_size):
        ypred = model(xb, x2b)
        to_append = [tf.argmax(x) for x in ypred]
        pred = pred + to_append
        true_append = [x for x in yb]
        true = true + true_append

    cf = tf.confusion_matrix(labels=true, predictions=pred)
    cf = np.array(cf)

    print(cf)

    plot_confusion_matrix(cf, class_dict)

    sys.exit(0)

