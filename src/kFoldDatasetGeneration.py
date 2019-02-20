import PickleGenerator
from pydub import AudioSegment
import os
from PickleGenerator import partition_dataset

#function to edit to correctly split the ds for the kFold
def splitDataset(datapath, trainpath, testpath, k=5):
    
# =============================================================================
# Cycle on all of the classes
# ============================================================================

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
    # Allocate the predefined portions of audio to different files
    # =============================================================================
                testSize = int(len(audio)/k)
                #testPortions.append(audio[:testSize])
                #trainPortions.append(audio[testSize:])
                fold = 0
                for i in range(0,testSize*k,testSize):
                    testTrack = audio[i:i+testSize]
                    trainTrack = audio[0:i] + audio[i+testSize:len(audio)]

                

    # =============================================================================
    # Create the new audio files and the folders to save them in
    # =============================================================================
                    out_trainpath = os.path.join(trainpath + str(fold), str(cat))
                    out_testpath = os.path.join(testpath + str(fold), str(cat))

                    if not os.path.isdir(out_trainpath):
                        os.makedirs(out_trainpath)
                    
                    trainTrack.export(os.path.join(out_trainpath, track), format=form) # export training partition
                    
                    if not os.path.isdir(out_testpath):
                        os.makedirs(out_testpath)
                    
                    testTrack.export(os.path.join(out_testpath, track), format=form) # export test partition

                    fold += 1

        return

if __name__ == "__main__":
        
        print("Splitting datatset for k-fold...")
        splitDataset("../dataset/5Classes", "../dataset/kFoldDataset/partitions/training", "../dataset/kFoldDataset/partitions/testing", 5)
        print("Dataset generated.")

        IN_PATH_TRAIN = "../dataset/kFoldDataset/partitions/training"
        IN_PATH_TEST = "../dataset/kFoldDataset/partitions/testing"
        OUT_PATH_TRAIN = "../dataset/kFoldDataset/segments/training"
        OUT_PATH_TEST = "../dataset/kFoldDataset/segments/testing"
        AUDIO_MS = 950
        HOP_MS = 475
        k = 5

        for i in range(0,k):
            partition_dataset(IN_PATH_TRAIN + str(i), OUT_PATH_TRAIN + str(i), AUDIO_MS, HOP_MS)
            partition_dataset(IN_PATH_TEST + str(i), OUT_PATH_TEST + str(i), AUDIO_MS, HOP_MS)

    