from rawnet import rawCNN
from spectronet import SpectroCNN
from DSEvidence import DSEvidence

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from Waver import Waver
from Spectrum import Spectrum

import pickle
import os

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


def gen_dataset(src_path, class_dict):
    def _read_aux(path, one_hot):
        ret = []
        if (not os.path.isdir(path)) and path.endswith('.wav'):
            val = ((Waver.get_waveform(path), Spectrum.compute_specgram_and_delta(path)), one_hot)
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
    Z = []
    Y = []
    for x,y in data:
        X.append(x[0])
        Z.append(x[1])
        Y.append(y)
    return X, Z, Y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    raw_path = "../models/rawNet.h5"
    spectro_path = "../models/spectroNet.h5"

    test_dataset_path = '../dataset/DSE_pickle'
    test_dataset_path_get = '../dataset/segments/testing'

    batch_size = 1 #nun me tocca' se ci tieni alle mani

    def _DScnn(x, rawnet, spectronet):
        c1 = rawnet(x[0])
        c2 = spectronet(x[1])

        c1 = tf.nn.softmax(c1)
        c2 = tf.nn.softmax(c2)

        c1 = list(c1.numpy()[0])
        c2 = list(c2.numpy()[0])

        out = DSEvidence.get_joint_mass(c1, c2)
        out = tf.convert_to_tensor(out, dtype=tf.float32)
        return tf.reshape(out, shape = [1, -1])

    class_train_dict, _ = load('../dataset/data_train_pickle')

    test_set = load(test_dataset_path)
    if test_set is None:
        print("No Test data")
        test_set = gen_dataset(test_dataset_path_get, class_train_dict)
        save(test_set, test_dataset_path)

    class_test_dict, test_data = test_set

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
        print("\rBatch num: " + str(batch), end='')
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


