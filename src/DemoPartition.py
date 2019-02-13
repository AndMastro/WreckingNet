import os

from pydub import AudioSegment
import numpy as np


def partition_track(track_path, out_path, ms, hop=None, get_drop=False):

    if hop is None:
        hop = ms

    audio = AudioSegment.from_file(track_path)

    size = len(audio)
    segments = []
    i = 0
    while i + ms < size:
        next_segment = audio[i:i + ms]
        segments.append(next_segment)
        i = i + hop

    form = track_path.split(".")[-1]

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    rms = np.array([x.rms for x in segments])
    q1 = np.percentile(rms, .25)
    q3 = np.percentile(rms, .75)

    lf = q1 - 1.5*(q3-q1)

    lo_sil = AudioSegment.empty()

    for idx, segment in enumerate(segments):
        if segment.rms < lf:
            lo_sil = lo_sil + segment
        else:
            segment.export(os.path.join(out_path, str(str(idx) + "." + form)), format=form)

    if get_drop:
        lo_sil.export(os.path.join(out_path, 'lo_sil.wav'), format=form)


def partition_dataset(in_path, out_path, ms, hop):
    if not os.path.isdir(in_path):
        print(out_path)
        partition_track(in_path, str(out_path), ms, hop)

    else:
        folders = os.listdir(in_path)
        for folder in folders:
            new_in_path = os.path.join(in_path, str(folder))
            new_out_path = os.path.join(out_path, str(folder))
            partition_dataset(new_in_path, new_out_path, ms, hop)


if __name__ == "__main__":
    import sys
    from splitDataset import split_datasets

    DATAPATH = "../dataset/5Classes"
    TRAINSEG = "../dataset/partitions/training"
    TESTSEG = "../dataset/partitions/testing"

    TrainOUT = "../dataset/segments/training"
    TestOUT = "../dataset/segments/testing"
    AUDIOMS = 30
    HOPMS = 15

    split_datasets(DATAPATH, TRAINSEG, TESTSEG)
    partition_dataset(TRAINSEG, TrainOUT, AUDIOMS, HOPMS)
    partition_dataset(TESTSEG, TestOUT, AUDIOMS, HOPMS)
    sys.exit(0)
