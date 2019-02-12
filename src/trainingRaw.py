import glob
import os
import random
import pickle

from rawnet import rawCNN
from Spectrum import Spectrum

import tensorflow as tf
import tensorflow.contrib.eager as tfe

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

    dataset_path = '../dataset/waveforms'
    dataset_path_get = '../dataset/segments'

    batch_size = 16
    epochs = 20

    # read data
    dataset = load(dataset_path)
    if dataset is None:
        print('None data')
        dataset = read_dataset(dataset_path_get)
        save(dataset, dataset_path)

    class_dict, data = dataset
    random.shuffle(data)

    X, Y = get_samples_and_labels(data)

    print(len(X))
    split_size = int(len(X)*0.7)

    Xtrain = X[:split_size]
    Ytrain = Y[:split_size]
    Xtest = X[split_size:]
    Ytest = Y[split_size:]

    Xtrain = tf.convert_to_tensor(Xtrain, dtype=tf.float32)
    Ytrain = tf.convert_to_tensor(Ytrain, dtype=tf.float32)
    Xtest = tf.convert_to_tensor(Xtest, dtype=tf.float32)
    Ytest = tf.convert_to_tensor(Ytest, dtype=tf.float32)
    train_it = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
    test_it = tf.data.Dataset.from_tensor_slices((Xtest, Ytest))


    def _parse_example(x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.int32)
        return x, y

    train_it = train_it.map(_parse_example)
    test_it = test_it.map(_parse_example)

    cnn = rawCNN()


    def loss(net, x, y):
        return tf.losses.sparse_softmax_cross_entropy(logits=net(x, training=True), labels=y)

    opt = tf.train.AdamOptimizer(learning_rate=0.000001) #it helps to go out the local minimum. with 0.0001 and 40 epochs ok. Mus remove oscillations.


    trainAcc = []
    testAcc = []
    lossValues = []

    for epoch in range(epochs):

        accTrain = tfe.metrics.SparseAccuracy()
        for xb, yb in train_it.batch(batch_size):
            ypred = cnn(xb)
            accTrain(predictions=ypred, labels=yb)
            lossValue = tf.losses.sparse_softmax_cross_entropy(logits=ypred, labels=yb)
            lossValues.append(lossValue)

        accTest = tfe.metrics.SparseAccuracy()
        for xb, yb in test_it.batch(batch_size):
            # print("test")
            # print(xb, yb)
            ypred = cnn(xb)
            accTest(predictions=ypred, labels=yb)

        trainAcc.append(accTrain.result().numpy())
        testAcc.append(accTest.result().numpy())
        print('==================================')
        print('Train accuracy at epoch {} is {} %'.format(epoch, trainAcc[-1] * 100))
        print('Test accuracy at epoch {} is {} %'.format(epoch, testAcc[-1] * 100))
        print('Loss value at epoch {} is {}'.format(epoch, lossValue))

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

    