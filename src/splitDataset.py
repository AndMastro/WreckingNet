# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 09:57:06 2019

@author: Ale
"""

from pydub import AudioSegment
import os

DATAPATH = "../dataset/UtahAudioData"
TRAINSEG = "../dataset/partitions/training"
TESTSEG = "../dataset/partitions/testing"


def split_datasets(datapath, trainpath, testpath, perc=0.7):
    
    perc = min(1.0, perc)
    
    cats = os.listdir(datapath)
    for cat in cats:
        print(cat)
        dirpath = os.path.join(datapath, str(cat))
        for track in os.listdir(dirpath):
            trackpath = os.path.join(dirpath, str(track))
            audio = AudioSegment.from_file(trackpath)
            
            form = track.split(".")[-1]

            train_seg = audio[:int(perc*len(audio))]
            test_seg = audio[int(perc*len(audio)):]

            out_trainpath = os.path.join(trainpath, str(cat))
            out_testpath = os.path.join(testpath, str(cat))

            if not os.path.isdir(out_trainpath):
                os.makedirs(out_trainpath)
            
            train_seg.export(os.path.join(out_trainpath, track), format=form)
            
            if not os.path.isdir(out_testpath):
                os.makedirs(out_testpath)
            
            test_seg.export(os.path.join(out_testpath, track), format=form)

    return
