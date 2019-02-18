import os
import json
import shutil
import random
import numpy as np

from pydub import AudioSegment
from Waver import Waver
from Spectrum import Spectrum
from utils import save

dataPath = "../dataset/5Classes"
percSplit = 0.7
audioMS = 30
audioHop = 15


def partition_track(track_path, out_path, ms, hop=None, get_drop=False):
    """
    :param track_path: str
        path of the track to partition
    :param out_path: str
        directory in which save the chuncks, if it do not exist it is created
    :param ms: int
        duration of the chunk to generate
    :param hop: int
        step to start next chunck
    :param get_drop: bool
        generate a track with the discarded audio
    :return: None
    """

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
    """
    :param in_path: str
        root of the dataset where to start
    :param out_path: str
        root of the new dataset
    :param ms: int
        length of the chuncks
    :param hop: int
        ms after start next chunck
    :return:
    """
    if not os.path.isdir(in_path):
        print(out_path)
        partition_track(in_path, str(out_path), ms, hop)

    else:
        folders = os.listdir(in_path)
        for folder in folders:
            new_in_path = os.path.join(in_path, str(folder))
            new_out_path = os.path.join(out_path, str(folder))
            partition_dataset(new_in_path, new_out_path, ms, hop)


def gen_dataset(src_path, class_dict=None):
    """
    :param src_path: STR
        root of the dataset to read
    :param class_dict: dict
        dict with the correspondence name-labels
    :return: dict, Object
        return a dict with the classes correspondence,
        and a list of (Objects, label)
    """

    def _read_aux(path, label):
        ret = []
        if (not os.path.isdir(path)) and path.endswith('.wav'):
            val = (Waver.get_waveform(path), Spectrum.compute_specgram_and_delta(path), label)
            ret.append(val)
        elif os.path.isdir(path):
            folders = os.listdir(path)
            for folder in folders:
                ret += _read_aux(os.path.join(path, str(folder)), label)
        return ret

    if class_dict is None:
        ret_dict = {}
        class_id = -1
    else:
        ret_dict = class_dict

    classes = os.listdir(src_path)
    dataset = []
    for class_type in classes:
        if class_dict is None:
            class_id += 1
            ret_dict[class_type] = class_id
        else:
            class_id = class_dict[class_type]
        print(class_type)
        new_data = _read_aux(os.path.join(src_path, class_type), class_id)
        random.shuffle(new_data)
        print('class size is: ', len(new_data))
        dataset = dataset+new_data

    random.shuffle(dataset)
    return ret_dict, dataset


def get_samples_and_labels(data):
    """
    :param data: data to unpack
    :return: X, Z, Y:
        X - list of objects to classify
        Z - secondary list of objects to classify
        Y - list of labels of the objects
    """
    X = []
    Z = []
    Y = []
    for x, z, y in data:
        X.append(x)
        Z.append(z)
        Y.append(y)
    return X, Z, Y


def generate_config(config_path='config.json', dataset_path=dataPath, percentage=percSplit, audio_ms=audioMS, audio_hop=audioHop, overwrite=False):
    """
    :param config_path: str
        path to config file
    :param dataset_path: str
        path to dataset root
    :param percentage:

    :param audio_ms:
    :param audio_hop:
    :param overwrite: bool
        whether to overwrite the old config with the new one
    :return: dict
        param dict
    """
    params = dict()

    try:
        with open(config_path, mode = 'r', encoding='utf-8') as fin:
            params = json.load(fin)
    except Exception as e:
        print(e)
        params['DATA_PATH'] = dataset_path
        params['PERCENTAGE'] = percentage
        params['AUDIO_MS'] = audio_ms
        params['HOP_MS'] = audio_hop

    params['SEG_ROOT'] = "../dataset/partitions" + str(int(params['PERCENTAGE'] * 100))
    params['TRAIN_SEG'] = params['SEG_ROOT'] + "/training"
    params['TEST_SEG'] = params['SEG_ROOT'] + "/testing"

    params['OUT_ROOT'] = "../dataset/segments_ms" + str(params['AUDIO_MS']) + "_hop" + str(params['HOP_MS'])
    params['TRAIN_OUT'] = params['OUT_ROOT'] + "/training"
    params['TEST_OUT'] = params['OUT_ROOT'] + "/testing"

    params['PICKLES_FOLDER'] = "../dataset/pickles/ms" + str(params['AUDIO_MS']) + "_hop" + str(params['HOP_MS'])
    params['TRAIN_PICKLE'] = params['PICKLES_FOLDER'] + "/train.p"
    params['TEST_PICKLE'] = params['PICKLES_FOLDER'] + "/test.p"
    params['DICT_JSON'] = params['PICKLES_FOLDER'] + "/classes.json"

    if overwrite:
        with open(config_path, mode='w+', encoding='utf-8') as fout:
            json.dump(params, fout)

    return params


if __name__ == "__main__":
    import sys
    from utils import split_datasets

    params = generate_config(overwrite=True)

    DATA_PATH = params['DATA_PATH']
    PERCENTAGE = params['PERCENTAGE']
    AUDIO_MS = params['AUDIO_MS']
    HOP_MS = params['HOP_MS']

    SEG_ROOT = params['SEG_ROOT']
    TRAIN_SEG = params['TRAIN_SEG']
    TEST_SEG = params['TEST_SEG']

    OUT_ROOT = params['OUT_ROOT']
    TRAIN_OUT = params['TRAIN_OUT']
    TEST_OUT = params['TEST_OUT']

    PICKLES_FOLDER = params['PICKLES_FOLDER']
    TRAIN_PICKLE = params['TRAIN_PICKLE']
    TEST_PICKLE = params['TEST_PICKLE']
    DICT_JSON = params['DICT_JSON']

    with open("config.json", mode='w+', encoding='utf-8') as fout:
        json.dump(params, fout)

    # check whether we have to regenerate them
    if os.path.exists(TRAIN_PICKLE) and os.path.exists(TEST_PICKLE) and os.path.exists(DICT_JSON):
        print("Data already exist, if you wanted to change try one or all of this: \n"
              "\t- delete config.json\n"
              "\t- delete one of: "+ str(TRAIN_PICKLE)+ ", "+ str(TEST_PICKLE)+", "+ str(DICT_JSON) + "\n"
              "then run me again")
        sys.exit(0)
    else:
        print("Generating data for: ")
        print("Percentage: ", params['PERCENTAGE'])
        print("MS: ", params['AUDIO_MS'])
        print("HOP: ", params['HOP_MS'])

    # this part is quite fast we can afford to do every time
    # we will treat the folders as tmp
    split_datasets(DATA_PATH, TRAIN_SEG, TEST_SEG, perc=PERCENTAGE)
    partition_dataset(TRAIN_SEG, TRAIN_OUT, AUDIO_MS, HOP_MS)
    partition_dataset(TEST_SEG, TEST_OUT, AUDIO_MS, HOP_MS)

    # generate the pickle pickle pickle yeah
    classes_dict, train_data = gen_dataset(TRAIN_OUT)

    if not os.path.isdir(PICKLES_FOLDER):
        os.makedirs(PICKLES_FOLDER)

    save(train_data, TRAIN_PICKLE)
    with open(DICT_JSON, mode='w+', encoding='utf-8') as fout:
        json.dump(classes_dict, fout)

    classes_dict, test_data = gen_dataset(TEST_OUT, classes_dict)
    save(test_data, TEST_PICKLE)

    # delete temp folders
    print("Deleting folder: ", SEG_ROOT)
    shutil.rmtree(SEG_ROOT)  # TRAIN_SEG, TEST_SEG
    print("Deleting folder: ", OUT_ROOT)
    shutil.rmtree(OUT_ROOT)  # TRAIN_OUT, TEST_OUT

    sys.exit(0)
