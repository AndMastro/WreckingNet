import os

from pydub import AudioSegment
from Spectrum import Spectrum


def partition_track(track_path, out_path, ms, hop=None, threshold=1000):

    if hop is None:
        hop = ms

    audio = AudioSegment.from_file(track_path)

    size = len(audio)
    segments = []
    i = 0
    while (i + ms < size):
        next_segment = audio[i:i + ms]
        segments.append(next_segment)
        i = i + hop

    form = track_path.split(".")[-1]

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # =============================
    #threshold = 0
    #x = 0
    #for el in segments:
    #    threshold = threshold + el.rms
    #    x = x+1

    #threshold = threshold / x
    # =============================
    sil = AudioSegment.empty()

    for idx, segment in enumerate(segments):
        if segment.rms < threshold:
            sil = sil + segment
            continue
        segment.export(os.path.join(out_path, str(str(idx) + "." + form)), format=form)

    sil.export(os.path.join(out_path, 'sil.wav'), format=form)


def partition_dataset(in_path, out_path, ms, hop):
    if not os.path.isdir(in_path):
        print(out_path)
        # in_path is a file, create out_path and convert
        partition_track(in_path, str(out_path), ms, hop)

        #for segment in os.listdir(out_path):
        #    seg_path = os.path.join(out_path, segment)
        #    Spectrum.get_specgram_librosa(seg_path)
    else:
        folders = os.listdir(in_path)
        for folder in folders:
            new_in_path = os.path.join(in_path, str(folder))
            new_out_path = os.path.join(out_path, str(folder))
            partition_dataset(new_in_path, new_out_path, ms, hop)
        # enumerate dirs
        # call partition


if __name__ == "__main__":
    import sys
    DATAPATH = "../dataset/5Classes"
    OUT = "../dataset/segments"
    AUDIOMS = 1000
    HOPMS = 500
    partition_dataset(DATAPATH, OUT, AUDIOMS, HOPMS)
    sys.exit(0)
