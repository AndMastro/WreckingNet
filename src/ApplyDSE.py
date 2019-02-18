import sys
import json
import random

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from rawnet import rawCNN
from spectronet import SpectroCNN
from DSEvidence import DSEvidence

from DemoPartition import get_samples_and_labels
from utils import get_class_numbers, get_reduced_set, load

tf.enable_eager_execution()

BATCH_SIZE = 128

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        with open('config.json', mode='r', encoding='utf-8') as fin:
            params = json.load(fin)
    except Exception as e:
        print(e)
        print("No config file, aborting...")
        sys.exit(0)

    raw_path = params['RAW_MODEL_PATH']
    spectro_path = params['SPECTRUM_MODEL_PATH']

    test_dataset_path = params['TEST_PICKLE']

    batch_size = BATCH_SIZE

    try:
        with open(params['DICT_JSON'], mode='r', encoding='utf-8') as fin:
            class_dict = json.load(fin)
    except Exception as e:
        print(e)
        print("We have lost correpsondances, aborting...")
        sys.exit(0)

    test_set = load(test_dataset_path)
    if test_set is None:
        print("No Test data, aborting...")
        sys.exit(0)

    random.shuffle(test_set)
    test_lens = get_class_numbers(test_set, class_dict)
    test_data = get_reduced_set(test_set, test_lens, 'min')


    def _DScnn(x, rawnet, spectronet):
        c1 = rawnet(x[0])
        c2 = spectronet(x[1])

        c1 = tf.nn.softmax(c1)
        c2 = tf.nn.softmax(c2)

        out = DSEvidence.tf_get_joint_mass(c1, c2)

        return out


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

    print("Loading weights for the rawNet")
    rawnet.load_weights(raw_path)
    print("Done")
    print("============================")
    print("Loading weights for the spectroNet")
    spectronet.load_weights(spectro_path)
    print("Done")

    print("Initializing nets")
    cnn = lambda x: _DScnn(x, rawnet, spectronet)
    print("Done")


    def _parse_example(x, y, z):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        z = tf.cast(z, tf.int32)
        return x, y, z


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

    print('Test accuracy is {} %'.format(accTest.result().numpy() * 100))

    print("=================================")
    print("Generating confusion matrix")
    cf = tf.confusion_matrix(labels=true, predictions=pred)
    cf = np.array(cf)

    print(cf)

    plt.matshow(cf)
    plt.colorbar()
    plt.show()

    print(np.array(pred))
    print(np.array(true))

    sys.exit(0)
