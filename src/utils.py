# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 09:57:06 2019

@author: Ale
"""

from pydub import AudioSegment
import os

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