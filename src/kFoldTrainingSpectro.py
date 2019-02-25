import os
import sys
import json
import random

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from spectronet import SpectroCNN
from PickleGenerator import get_samples_and_labels
from utils import get_class_numbers, get_reduced_set, load, plot_confusion_matrix


BATCH_SIZE = 64  # 4096
EPOCHS = 20
LEARNING_RATE = 0.0005

tf.enable_eager_execution()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    k_fold_number = 2
    train_dataset_path = "../dataset/kFoldDataset/pickles/trainPickle" + str(k_fold_number)
    test_dataset_path = "../dataset/kFoldDataset/pickles/testPickle" + str(k_fold_number)
    model_path = "../models/kFold/modelSpectro"

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    model_path = os.path.join(model_path, str(k_fold_number)+'.h5')

    with open("../dataset/kFoldDataset/pickles/classes.json") as classesFile:
        class_dict = json.load(classesFile)

    batch_size = BATCH_SIZE
    epochs = EPOCHS
    learning_rate = LEARNING_RATE

    # read train data
    train_set = load(train_dataset_path)
    if train_set is None:
        print("No Train data, aborting...")
        sys.exit(0)

    # read train data
    test_set = load(test_dataset_path)
    if test_set is None:
        print("No Test data, aborting...")
        sys.exit(0)

    random.shuffle(train_set)
    random.shuffle(test_set)  # should be here

    train_lens = get_class_numbers(train_set, class_dict)
    train_data = get_reduced_set(train_set, train_lens, 'min')

    test_lens = get_class_numbers(test_set, class_dict)
    test_data = get_reduced_set(test_set, test_lens, 'min')

    _, Xtrain, Ytrain = get_samples_and_labels(train_data)
    _, Xtest, Ytest = get_samples_and_labels(test_data)

    print("Train size", len(Ytrain))
    print("Test size", len(Ytest))

    Xtrain = tf.convert_to_tensor(Xtrain, dtype=tf.float32)
    Ytrain = tf.convert_to_tensor(Ytrain, dtype=tf.float32)
    Xtest = tf.convert_to_tensor(Xtest, dtype=tf.float32)
    Ytest = tf.convert_to_tensor(Ytest, dtype=tf.float32)

    print("Allocating tensors")

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

    plot_confusion_matrix(cf, class_dict)

    cnn.save_weights(model_path)

    sys.exit(0)
