import DemoPartition
from pydub import AudioSegment
import os

#function to edit to correctly split the ds for the kFold
def splitDataset(datapath, trainpath, testpath, k=5):
    
# =============================================================================
# Cycle on all of the classes
# =============================================================================
    for i in range(0, k):
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

#if __name__ == "__main__":
    