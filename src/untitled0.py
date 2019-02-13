# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 09:57:06 2019

@author: Ale
"""

from pydub import AudioSegment
import os

DATAPATH = "../dataset/UtahAudioData"
TRAINSEG = "../dataset/partitions/training"
TESTSEG  = "../dataset/partitions/testing"

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
            
            train_seg = audio[:int(round(perc*len(audio), 0))]
            test_seg = audio[int(round(perc*len(audio)))+1:]
            
            if not os.path.isdir(trainpath):
                os.makedirs(trainpath)
            
            train_seg.export(os.path.join(trainpath, track), format=form)
            
            if not os.path.isdir(testpath):
                os.makedirs(testpath)
            
            test_seg.export(os.path.join(testpath, track), format=form)
    
    
    return
    

if __name__ == "__main__":
    split_datasets(DATAPATH, TRAINSEG, TESTSEG, 0.7)
