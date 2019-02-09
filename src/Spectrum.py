import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display

import os

class Spectrum:

    @staticmethod
    def plot_spectrogram(rate, data, NFFT=1024, noverlap=256):
        """
        :param rate: int
            Sample rate of the audio file
        :param data: numpy array
            Data to plot
        :param NFFT: int
            The number of data points used in each block for the FFT.
        :param noverlap: int
         The number of points of overlap between blocks
        :return: Figure
        """
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.specgram(data, NFFT=NFFT, noverlap=noverlap, Fs=rate)
        ax.axis('off')
        return fig

    @staticmethod
    def get_specgram_mplib(path, fmt="svg", NFFT=1024, noverlap=512):
        """
        :param path: str
            path to the .wav file to generate spectrogram
        :param fmt: str
            name of the format to save the plot, default svg
            the format available are the same supported by matplotlib.pyplot.savefig
        :param NFFT: int
            The number of data points used in each block for the FFT.
        :param noverlap: int
            The number of points of overlap between blocks
        :return: None
            creates a image into the path with the same name of input file
            If the data has more channels, creates more svg files.
        """
        out_name = path
        if path.endswith(".wav"):
            out_name = path[:-4]
        rate, data = wavfile.read(path)
        if data.ndim > 1:
            # we have more than one channel, we have to do more plots
            for i in range(0, data.ndim):
                dimension_data = data[:,i]
                out_path = out_name + "_channel_" + str(i) + "." + fmt
                fig = Spectrum.plot_spectrogram(rate, dimension_data, NFFT, noverlap);
                fig.savefig(out_path, format=fmt, frameon='false')
        else:
            out_name += "." + fmt
            fig = Spectrum.plot_spectrogram(rate, data, NFFT, noverlap);
            fig.savefig(out_name, format=fmt, frameon='false')

    @staticmethod
    def get_specgram_librosa(path, fmt='svg', sample_rate=2205, nfft=1024, hop_len=512, n_mel_bands=60):
        """
        :param path: str
            path where the wav file is located
        :param fmt: str
            format to save the image
        :param sample_rate:
            sample rate to re-sample the wav.
            default is set to audible frequencies
        :param nfft: int
            The number of data points used in each block for the FFT.
        :param hop_len: int
            The number of points of overlap between blocks
        :param n_mel_bands:
            number of Mel bands to generate
        :return: None
            creates a image into the path with the same name of input file
        """
        out_name = path
        if path.endswith(".wav"):
            out_name = path[:-4]+"."+ fmt
        sig, fs = librosa.load(path, sr=sample_rate)
        # missing signal normalization
        # generating first channel, log-scaled mel spectrogram (default parameters are the one used in the paper)
        S = librosa.feature.melspectrogram(y=sig, sr=fs, n_fft=nfft, hop_length=hop_len, n_mels=n_mel_bands)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

        # delta computation (librosa.feature.delta)

        plt.savefig(out_name, format=fmt, frameon='false', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    DATAPATH = "../dataset/segments"
    folders = os.listdir(DATAPATH)
    for folder in folders:
        dirpath = os.path.join(DATAPATH, str(folder))
        for track in os.listdir(dirpath):
            trackpath = os.path.join(dirpath, str(track))
            for segment in os.listdir(trackpath):
                segpath = os.path.join(trackpath, segment)
                Spectrum.get_specgram_librosa(segpath, 'png')
