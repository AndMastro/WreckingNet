# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:51:28 2019

@author: Ale
"""

import json
import os
import sys
import random
import shutil

import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

import PickleGenerator
from rawnet import rawCNN
from spectronet import SpectroCNN
from DSEvidence import DSEvidence

from Spectrum import Spectrum
from Waver import Waver

from PickleGenerator import get_samples_and_labels
from utils import get_class_numbers, get_reduced_set, load, plot_confusion_matrix


CLASSFILE   = "../dataset/kFoldDataset/pickles/classes.json"
MODELRDIR   = r"../models/kFold/modelRaw/4.h5"
MODELSDIR   = r"../models/kFold/modelSpectro/4.h5"


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
    
    print(os.path.exists(MODELSDIR))
    
    rawnet.load_weights(MODELRDIR)
    spectronet.load_weights(MODELSDIR)

    cnn = lambda x: _DScnn(x, rawnet, spectronet)

    test_it = test_it.map(_parse_example)

    pred = []
    true = []

    print("Testing...")
    
    classes = {}
    
    batch = 0
    
    for x, z, yb in test_it.batch(1):
        ypred = cnn((x, z))         
        to_append = [tf.argmax(x) for x in ypred]
        to_append = to_append[0].numpy()
        
        if to_append in classes:
            classes[to_append] += 1
        else:
            classes[to_append] = 1
            
        if batch%1000 == 0:
            print("Batch num: " + str(batch), end='\r', flush=True)
            
        batch += 1
    
    print(classes)
    
    audio_class = -1
    audio_max = 0
    
    for c in classes:
        if classes[c] > audio_max:
            audio_class = c
            audio_max = classes[c]
    
    print("Audio class:", audio_class)

    return audio_class
    
def get_data(PICKLEDIC, PICKLENAME):
    
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
    
    rawdata, specdata, _ = get_samples_and_labels(test_data)  
    
    return rawdata, specdata
    

if __name__ == "__main__":

    audiopath = '../dataset/predict/ConcreteMixer_onsite.wav'
    tmp_segments = '../dataset/predict/chuncks'

    PickleGenerator.partition_track(audiopath, tmp_segments, 30, 15)

    raw = []
    specs = []
    # generate spectrum and raw data
    for file in os.listdir(tmp_segments):
        path = os.path.join(tmp_segments, file)
        val = (Waver.get_waveform(path), Spectrum.compute_specgram_and_delta(path))
        raw.append(val[0])
        specs.append(val[1])

    shutil.rmtree(tmp_segments)

    classes = json.load(CLASSFILE)
    x = predict([raw, specs])
    for k in classes:
        if classes[k] == x:
            print("Class is:", k)
