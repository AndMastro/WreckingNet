from rawnet import rawCNN
from spectronet import SpectroCNN
from DSEvidence import DSEvidence

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

if __name__ == "__main__":
    raw_path = "../models/rawNet.h5"
    spectro_path = "../models/spectroNet.h5"

    rawnet = rawCNN()
    spectronet = SpectroCNN()

    print("Hello biatch")