import sys
import os
import random
import pickle

from spectronet import SpectroCNN
from Spectrum import Spectrum
from utils import get_class_numbers, get_reduced_set

import tensorflow as tf
import tensorflow.contrib.eager as tfe

BATCH_SIZE = 4096
EPOCHS = 20
LEARNING_RATE = 0.001

tf.enable_eager_execution()


def load(path):
    try:
        with open(path, 'rb') as fin:
            dataset = pickle.load(fin)
    except Exception as e:
        print(e)
        dataset = None
    return dataset


def save(dataset, path):
    with open(path, 'wb') as fout:
        pickle.dump(dataset, fout)


def read_dataset(src_path):
    def _read_aux(path, one_hot):
        ret = []
        if (not os.path.isdir(path)) and path.endswith('.wav'):
            val = (Spectrum.compute_specgram_and_delta(path), one_hot)
            ret.append(val)
        elif os.path.isdir(path):
            folders = os.listdir(path)
            for folder in folders:
                ret += _read_aux(os.path.join(path, str(folder)), one_hot)
        return ret

    classes = os.listdir(src_path)
    class_dict = dict()
    class_id = 0
    dataset = []
    for class_type in classes:
        print(class_type)
        new_data = _read_aux(os.path.join(src_path, class_type), class_id)
        print('class size is: ', len(new_data))
        dataset = dataset+new_data
        class_dict[class_type] = class_id
        class_id = class_id + 1

    return class_dict, dataset


def read_dataset_test(src_path, class_dict):
    def _read_aux(path, one_hot):
        ret = []
        if (not os.path.isdir(path)) and path.endswith('.wav'):
            val = (Spectrum.compute_specgram_and_delta(path), one_hot)
            ret.append(val)
        elif os.path.isdir(path):
            folders = os.listdir(path)
            for folder in folders:
                ret += _read_aux(os.path.join(path, str(folder)), one_hot)
        return ret

    classes = os.listdir(src_path)
    dataset = []
    for class_type in classes:
        class_id = class_dict[class_type]
        print(class_type)
        new_data = _read_aux(os.path.join(src_path, class_type), class_id)
        print('class size is: ', len(new_data))
        dataset = dataset+new_data

    return class_dict, dataset


def get_samples_and_labels(data):
    X = []
    Y = []
    for x,y in data:
        X.append(x)
        Y.append(y)
    return X, Y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    model_path = "../models/spectroNet.h5"

    train_dataset_path = '../dataset/data_train_pickle'
    train_dataset_path_get = '../dataset/segments/training'

    test_dataset_path = '../dataset/data_test_pickle'
    test_dataset_path_get = '../dataset/segments/testing'

    batch_size = BATCH_SIZE
    epochs = EPOCHS
    learning_rate = LEARNING_RATE

    # read train data
    train_set = load(train_dataset_path)
    if train_set is None:
        print("No Train data")
        train_set = read_dataset(train_dataset_path_get)
        save(train_set, train_dataset_path )

    class_train_dict, train_data = train_set
    train_lens = get_class_numbers(train_data, class_train_dict)
    train_data = get_reduced_set(train_data, train_lens, 'min')
    random.shuffle(train_data)

    # read train data
    test_set = load(test_dataset_path)
    if test_set is None:
        print("No Test data")
        test_set = read_dataset_test(test_dataset_path_get, class_train_dict)
        save(test_set, test_dataset_path)

    class_test_dict, test_data = test_set
    random.shuffle(test_data)
    test_lens = get_class_numbers(test_data, class_test_dict)
    test_data = get_reduced_set(test_data, test_lens, 'min')


    Xtrain, Ytrain = get_samples_and_labels(train_data)
    Xtest, Ytest = get_samples_and_labels(test_data)

    print("Train size", len(Ytrain))
    print("Test size", len(Ytest))

    Xtrain = tf.convert_to_tensor(Xtrain, dtype=tf.float32)
    Ytrain = tf.convert_to_tensor(Ytrain, dtype=tf.float32)
    Xtest = tf.convert_to_tensor(Xtest, dtype=tf.float32)
    Ytest = tf.convert_to_tensor(Ytest, dtype=tf.float32)

    print("Allocationg tensors")

    train_it = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
    test_it = tf.data.Dataset.from_tensor_slices((Xtest, Ytest))

    print("Done")

    def _parse_example(x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.int32)
        return x, y

    train_it = train_it.map(_parse_example)
    test_it = test_it.map(_parse_example)

    cnn = SpectroCNN()


    def loss(net, x, y):
        return tf.losses.sparse_softmax_cross_entropy(logits=net(x, training=True), labels=y)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    trainAcc = []
    testAcc = []
    lossValues = []

    print("Train Loop")

    for epoch in range(epochs):

        accTrain = tfe.metrics.SparseAccuracy()
        for xb, yb in train_it.batch(batch_size):
            ypred = cnn(xb)
            accTrain(predictions=ypred, labels=yb)
            lossValue = tf.losses.sparse_softmax_cross_entropy(logits=ypred, labels=yb)
            lossValues.append(lossValue)

        accTest = tfe.metrics.SparseAccuracy()
        for xb, yb in test_it.batch(batch_size):
            ypred = cnn(xb)
            accTest(predictions=ypred, labels=yb)

        trainAcc.append(accTrain.result().numpy())
        testAcc.append(accTest.result().numpy())
        print('==================================')
        print('Train accuracy at epoch {} is {} %'.format(epoch, trainAcc[-1] * 100))
        print('Test accuracy at epoch {} is {} %'.format(epoch, testAcc[-1] * 100))
        print('Loss value at epoch {} is {}'.format(epoch, lossValue))
        print('==================================')

        for xb, yb in train_it.shuffle(1000).batch(batch_size):
            opt.minimize(lambda: loss(cnn, xb, yb))

    plt.plot(trainAcc)
    plt.show()
    plt.plot(testAcc)
    plt.show()
    plt.plot(lossValues)
    plt.show()

    pred = []
    true = []

    for xb, yb in test_it.batch(batch_size):
        ypred = cnn(xb)
        to_append = [tf.argmax(x) for x in ypred]
        pred = pred + to_append
        true_append = [x for x in yb]
        true = true + true_append

    cf = tf.confusion_matrix(labels=true, predictions=pred)
    cf = np.array(cf)

    print(cf)

    plt.matshow(cf)
    plt.colorbar()
    plt.show()

    print(np.array(pred))
    print(np.array(true))

    cnn.save_weights(model_path)

    sys.exit(0)