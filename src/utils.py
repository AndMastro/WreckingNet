# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 09:57:06 2019

@author: Ale
"""

import os
import pickle
import itertools

import numpy as np
import matplotlib.pyplot as plt

from pydub import AudioSegment


def split_datasets(datapath, trainpath, testpath, perc=0.7):
    
# =============================================================================
# The percentage of each video to allocate to training (the rest will be the test set)
# =============================================================================
    perc = min(1.0, perc)       # percentage may be 100% maximum
    
# =============================================================================
# Cycle on all of the classes
# =============================================================================
    cats = os.listdir(datapath)
    
    for cat in cats:
        print(cat)
        
        dirpath = os.path.join(datapath, str(cat))          # path of the single category

# =============================================================================
# Cycle on all the tracks of a category
# =============================================================================
        for track in os.listdir(dirpath):

# =============================================================================
# Extract the audio itself of the track with pydub
# =============================================================================
            trackpath = os.path.join(dirpath, str(track))
            audio = AudioSegment.from_file(trackpath)

# =============================================================================
# Save format of the file
# =============================================================================
            form = track.split(".")[-1]     # all track files follow the name.format archetipe

# =============================================================================
# Allocate the predefined portions of audio to two different files
# =============================================================================
            train_seg = audio[:int(perc*len(audio))]
            test_seg = audio[int(perc*len(audio)):]

# =============================================================================
# Create the new audio files and the folders to save them in
# =============================================================================
            out_trainpath = os.path.join(trainpath, str(cat))
            out_testpath = os.path.join(testpath, str(cat))

            if not os.path.isdir(out_trainpath):
                os.makedirs(out_trainpath)
            
            train_seg.export(os.path.join(out_trainpath, track), format=form) # export training partition
            
            if not os.path.isdir(out_testpath):
                os.makedirs(out_testpath)
            
            test_seg.export(os.path.join(out_testpath, track), format=form) # export test partition

    return


def partition_dataset(datapath, out, ms):
    
    cats = os.listdir(datapath)
    for cat in cats:
        dirpath = os.path.join(datapath, str(cat))
        for track in os.listdir(dirpath):
            trackpath = os.path.join(dirpath, str(track))
            audio = AudioSegment.from_file(trackpath)
            
            segments = [audio[(i*ms):((i+1)*ms)] for i in range(0, len(audio)//ms)]
            
            outpath = os.path.join(out, str(cat), str(track.split(".")[0]))
            
            if not os.path.isdir(outpath):
                os.makedirs(outpath)
                
            form = track.split(".")[-1]
                    
            for idx, segment in enumerate(segments):
                segment.export(os.path.join(outpath, str(str(idx) + "." + form)), format=form)


def get_reduced_set(data, lens, mode='min'):
    
    assert (mode in ['min', 'all']) or (type(mode) == int and mode > 0)
    
    new_data = []
    taken = {}
    for c in lens:
        taken[c] = 0
    
    if mode == 'min':        
        max_samples = min([lens[x] for x in lens])
    elif mode == 'all':
        max_samples = len(data)
    elif mode == 'avg':
        max_samples = sum(lens.values())/len(lens)
    else:
        max_samples = mode
    
    for d in data:        
        if taken[d[2]] >= max_samples:
            continue
        new_data.append(d)
        taken[d[2]] += 1
        
    return new_data


def get_class_numbers(data, classes):
    
    lens = {}
    
    inverse_classes= {}
    for c in classes:
        inverse_classes[classes[c]] = c
    
    for c in inverse_classes:
        lens[c] = 0
    
    for d in data:
        lens[d[2]] += 1
        
    return lens


def load(path):
    """
    :param path: str
        path where to load the dataset
    :return:
        return the dataset saved in the pickle file
        If the path doesn't exist return None
    """
    try:
        with open(path, 'rb') as fin:
            dataset = pickle.load(fin)
    except Exception as e:
        print(e)
        dataset = None
    return dataset


def save(dataset, path):
    """
    :param dataset: Any
        object to save inside the pickle file
    :param path:
        pathe where to create the pickle storing the object
    :return:
        None
    """
    with open(path, 'wb') as fout:
        pickle.dump(dataset, fout)


def get_samples_and_labels(data):
    """
    :param data: data to unpack
    :return: X, Y:
        X - list of objects to classify
        Y - list of labels of the objects
    """
    X = []
    Y = []
    for x, y in data:
        X.append(x)
        Y.append(y)
    return X, Y

def normalize_cf(cf):
    """
    :param cf: np.array
        confusion matrix to normalize
    :return: np.array
        the confusion matrix normalized
    """
    return cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]


def get_sorted_class_names(class_dict):
    """
    :param class_dict: dict
        dict keeping the mapping class_name-class_id (where class id is an integer or a float)
    :return: list
        list of class_names, sorted by class_id
    """
    l = []
    for k in class_dict:
        l.append((class_dict[k], k))
    l.sort()
    return [x[1] for x in l]


def plot_confusion_matrix(cm, class_dict,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Code adapted from sklearn documentation
    """

    if normalize:
        cm = normalize_cf(cm)

    classes = get_sorted_class_names(class_dict)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
