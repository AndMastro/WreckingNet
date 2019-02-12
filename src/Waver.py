# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 09:57:14 2019

@author: Ale
"""

TESTPATH = r"../dataset/waves/compactor_ingersroll_rand/IRCOM_onsite.wav"
DATAPATH = r"../dataset/segments"
SAVEPATH = r"../dataset/waveforms"
PICKFILE = r"../dataset/data_pickle"

import os
import pickle
import librosa
import librosa.display

class Waver:
    
    def get_waveform(path, sample_rate=22050, normalize=False):
        
        signal, _ = librosa.load(path, sr=sample_rate)
        if normalize:
            signal = librosa.util.normalize(signal)
        
        return signal
    
    def save_waves(datapath, outfile, classfile):
        class_dict = {}
        data = []
        
        class_dict, _ = pickle.load(open(classfile, 'rb'))
        
        for cat in os.listdir(datapath):
            catpath = os.path.join(datapath, cat)
            
            if cat not in class_dict:
                continue            
            
            curl = class_dict[cat]    
            
            print(curl)
            
            for track in os.listdir(catpath):                
                trackpath = os.path.join(catpath, track)  
                
                for segment in os.listdir(trackpath):                    
                    signal = Waver.get_waveform(os.path.join(trackpath, segment))
                    data.append((signal, curl))
        
        print("dumping...")
        dataset = (class_dict, data)  
        pickle.dump(dataset, open(outfile, 'wb'))
        print("dumped")


# =============================================================================
# if __name__ == "__main__":
#     Waver.save_waves(DATAPATH, SAVEPATH, PICKFILE)
# =============================================================================
