# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 20:41:42 2019

@author: Ale
"""

from pydub import AudioSegment
import os

DATAPATH = r"..\dataset\UtahAudioData"
OUT = r"..\dataset\segments"
AUDIOMS = 30000

def partition_dataset(datapath, out, ms):
    
    cats = os.listdir(datapath)
    for cat in cats:
        dirpath = os.path.join(DATAPATH, str(cat))
        for track in os.listdir(dirpath):
            trackpath = os.path.join(dirpath, str(track))
            audio = AudioSegment.from_file(trackpath)
            
            segments = [audio[i:i+ms] for i in range(len(audio)//ms)]
            
            outpath = os.path.join(out, str(cat), str(track.split(".")[0]))
            
            if not os.path.isdir(outpath):
                os.makedirs(outpath)
                
            form = track.split(".")[-1]
                    
            for idx, segment in enumerate(segments):
                segment.export(os.path.join(outpath, str(str(idx) + "." + form)), format=form)


if __name__ == "__main__":
    partition_dataset(DATAPATH, OUT, AUDIOMS)