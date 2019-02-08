# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 20:41:42 2019

@author: Ale
"""

from pydub import AudioSegment
import os

AUDIOPATH = r"..\..\Dataset\DEFINETLY NOT PORN\UtahAudioData\[New] Concrete Mixer 3\Reg on site\concretemixer3_onsite.wav"
OUT = "output"
AUDIOMS = 30000

# =============================================================================
# audio = os.listdir(AUDIOPATH)
# =============================================================================


audio = AudioSegment.from_file(AUDIOPATH)

segments = [audio[i:i+AUDIOMS] for i in range(len(audio)//AUDIOMS)]
if OUT not in os.listdir("."):
    os.mkdir(OUT)
    
for idx, segment in enumerate(segments):
    segment.export(str(OUT + "\segment" + str(idx) + ".wav"), format="wav")

