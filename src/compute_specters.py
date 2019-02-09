# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 18:07:53 2019

@author: Ale
"""

from audio_splitter import partition_dataset
import Spectrum
import os

DATAPATH = "../dataset/UtahAudioData"
OUT = "../dataset/segments"
AUDIOMS = 30000
SPECFORMAT = 'png'
SPECWHICH = 'log'

def main(datapath, outpath, ms, form, spec):
    
    print("Partitioning dataset...")    
    partition_dataset(datapath, outpath, ms)
    print("Audios partitioned into", str(ms), "long segments")
    
    print("\nCreating spectrograms...")
    folders = os.listdir(outpath)
    for folder in folders:
        print("Class:", folder)
        dirpath = os.path.join(outpath, str(folder))
        for idx, track in enumerate(os.listdir(dirpath)):
            print("Working on audio", str(idx+1), "of", str(len(os.listdir(dirpath))))
            trackpath = os.path.join(dirpath, str(track))
            for idx2, segment in enumerate(os.listdir(trackpath)):
                print("Working on segment", str(idx2+1), "of", str(len(os.listdir(trackpath)) - idx2))
                segpath = os.path.join(trackpath, segment)
                Spectrum.Spectrum.get_specgram_librosa(segpath, form, spec)
    print("Spectrograms created")
    
    return

if __name__ == "__main__":
    main(DATAPATH, OUT, AUDIOMS, SPECFORMAT, SPECWHICH)