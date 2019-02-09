# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 20:41:42 2019

@author: Ale
"""

from pydub import AudioSegment
import os

# =============================================================================
# DATAPATH = "../dataset/UtahAudioData"
# OUT = "../dataset/segments"
# AUDIOMS = 30000
# =============================================================================

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


# =============================================================================
# if __name__ == "__main__":
#     partition_dataset(DATAPATH, OUT, AUDIOMS)
# =============================================================================
