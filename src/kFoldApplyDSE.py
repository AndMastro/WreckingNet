import os
import sys
import json
import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from rawnet import rawCNN
from spectronet import SpectroCNN
from DSEvidence import DSEvidence

from PickleGenerator import get_samples_and_labels
from utils import get_class_numbers, get_reduced_set, load, plot_confusion_matrix

tf.enable_eager_execution()

BATCH_SIZE = 2048


def _DScnn(x, rawnet, spectronet):
    c1 = rawnet(x[0])
    c2 = spectronet(x[1])

    c1 = tf.nn.softmax(c1)
    c2 = tf.nn.softmax(c2)

    out = DSEvidence.tf_get_joint_mass(c1, c2)

    return out


def _parse_example(x, y, z):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    z = tf.cast(z, tf.int32)
    return x, y, z


if __name__ == "__main__":

    batch_size = BATCH_SIZE

    folds = 5
    train_path = "../dataset/kFoldDataset/pickles/trainPickle"
    test_path = "../dataset/kFoldDataset/pickles/testPickle"
    model_raw = "../models/kFold/modelRaw"
    model_spectro = "../models/kFold/modelSpectro"

    with open("../dataset/kFoldDataset/pickles/classes.json") as classesFile:
        class_dict = json.load(classesFile)

    ACC = []

    for i in range(folds):
        train_dataset_path = train_path + str(i)
        test_dataset_path = test_path + str(i)
        raw_path = os.path.join(model_raw, str(i) + '.h5')
        spectro_path = os.path.join(model_spectro, str(i) + '.h5')

        # read test data
        test_set = load(test_dataset_path)
        if test_set is None:
            print("No Test data, aborting...")
            sys.exit(0)

        random.shuffle(test_set)  # should be here

        test_lens = get_class_numbers(test_set, class_dict)
        test_data = get_reduced_set(test_set, test_lens, 'min')

        Xtest_raw, Xtest_spec, Ytest = get_samples_and_labels(test_data)

        size = len(Ytest)

        Xtest_raw = tf.convert_to_tensor(Xtest_raw, dtype=tf.float32)
        Xtest_spec = tf.convert_to_tensor(Xtest_spec, dtype=tf.float32)
        Ytest = tf.convert_to_tensor(Ytest, dtype=tf.float32)

        print("Allocating tensors")

        test_it = tf.data.Dataset.from_tensor_slices((Xtest_raw, Xtest_spec, Ytest))

        rawnet = rawCNN()
        spectronet = SpectroCNN()

        for x, z, yb in test_it.batch(1):
            print(x.shape)
            print(z.shape)
            _ = rawnet(x)
            _ = spectronet(z)
            break

        rawnet.load_weights(raw_path)
        spectronet.load_weights(spectro_path)

        cnn = lambda x: _DScnn(x, rawnet, spectronet)

        test_it = test_it.map(_parse_example)

        pred = []
        true = []

        print("Testing...")

        batch = 0
        accTest = tfe.metrics.SparseAccuracy()
        step = 0
        for x, z, yb in test_it.batch(batch_size):
            ypred = cnn((x, z))
            accTest(predictions=ypred, labels=yb)
            to_append = [tf.argmax(x) for x in ypred]
            pred = pred + to_append
            true_append = [x for x in yb]
            true = true + true_append
            print("Batch num: " + str(batch), end='\r', flush=True)
            batch += 1

        ACC.append(accTest.result().numpy())
        print(sum(ACC)/len(ACC))

        print("=================================")
        print("Generating confusion matrix")
        cf = tf.confusion_matrix(labels=true, predictions=pred)
        cf = np.array(cf)

        print(cf)

        plot_confusion_matrix(cf, class_dict)

    print(sum(ACC)/len(ACC))
