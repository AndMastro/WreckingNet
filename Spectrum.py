from scipy.io import wavfile
import matplotlib.pyplot as plt


class Spectrum:

    @staticmethod
    def get_data(path):
        """
        :param path: str, path to the .wav file to generate spectrogram
        :return: None, creates a .svg image into the path with the same name of input file
        """
        out_path = path
        if path.endswith(".wav"):
            out_path = path.replace(".wav", "")
        out_path += ".svg"
        rate, data = wavfile.read(path)
        if data.ndim > 1:
            # we have more than one channel
            data = data[:, 0]

        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.specgram(data, Fs=rate)
        ax.axis('off')
        fig.savefig(out_path, format='svg', frameon='false')
