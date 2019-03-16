# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:51:28 2019

@author: Ale
"""

import json
import os
import sys
import random

import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

from rawnet import rawCNN
from spectronet import SpectroCNN
from DSEvidence import DSEvidence

# =============================================================================
# from Spectrum import Spectrum
# from Waver import Waver
# =============================================================================
from PickleGenerator import get_samples_and_labels
from utils import get_class_numbers, get_reduced_set, load, plot_confusion_matrix

DATADIR     = r"..\dataset\5Classes"
CLASSDIR    = r"..\dataset\kFoldDataset\pickles"
CLASSFILE   = "classes.json"
MODELSDIR   = r"..\models\30"

PICKLEDIC   = r"..\dataset\kFoldDataset\pickles"
PICKLENAME  = "testPickle4"

CLASS = 0
BATCH = 1
AUDIO = 2

tf.enable_eager_execution()

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



def get_classes(path):
    
    classes = json.load(open(path, 'r'))
    return classes

def predict(segments):
    
    Xtest_raw = segments[0]
    Xtest_spec = segments[1]
    Ytest = np.full((len(Xtest_raw)), CLASS)
    
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

    rawnet.load_weights(os.path.join(MODELSDIR, 'raw.h5'))
    spectronet.load_weights(os.path.join(MODELSDIR, 'spectro.h5'))

    cnn = lambda x: _DScnn(x, rawnet, spectronet)

    test_it = test_it.map(_parse_example)

    pred = []
    true = []

    print("Testing...")
    
    classes = {}
    
    batch = 0
    accTest = tfe.metrics.SparseAccuracy()
    step = 0
    for x, z, yb in test_it.batch(1):
        ypred = cnn((x, z))
# =============================================================================
#         if ypred in classes:
#             classes[ypred] += 1
#         else:
#             classes[ypred] = 1
# =============================================================================
        print(ypred)
            
        accTest(predictions=ypred, labels=yb)
        to_append = [tf.argmax(x) for x in ypred]
        pred = pred + to_append
        true_append = [x for x in yb]
        true = true + true_append
        print("Batch num: " + str(batch), end='\r', flush=True)
        batch += 1

    


if __name__ == "__main__":
        
    dirpath = os.path.join(CLASSDIR, CLASSFILE)    
    classes = get_classes(dirpath)
    class_used = list(classes.keys())[CLASS]
    classpath = os.path.join(DATADIR, class_used)
    
    audioname = os.listdir(classpath)[min(len(os.listdir(classpath)), AUDIO)]    
    audiopath = os.path.join(classpath, audioname)
      
# =============================================================================
#     wave = Waver.get_waveform(audiopath)
#     specgram = Spectrum.compute_specgram_and_delta(audiopath)
# =============================================================================

    # read test data
    test_set = load(os.path.join(PICKLEDIC, PICKLENAME))
    if test_set is None:
        print("No Test data, aborting...")
        sys.exit(0)

    random.shuffle(test_set)  # should be here
    
    with open("../dataset/kFoldDataset/pickles/classes.json") as classesFile:
        class_dict = json.load(classesFile)

    test_lens = get_class_numbers(test_set, class_dict)
    test_data = get_reduced_set(test_set, test_lens, 'min')
    
    predict([test_data[0][0],test_data[0][1]])

    
