from scipy.io import wavfile
import matplotlib.pyplot as plt


class Spectrum:

    @staticmethod
    def plot_spectrogram(rate, data):
        """
        :param rate: int
            Sample rate of the audio file
        :param data: numpy array
            Data to plot
        :return: Figure
        """
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.specgram(data, Fs=rate)
        ax.axis('off')
        return fig

    @staticmethod
    def get_specgram(path, fmt="svg"):
        """
        :param path: str
            path to the .wav file to generate spectrogram
        :param fmt: str
            name of the format to save the plot, default svg
            the format available are the same supported by matplotlib.pyplot.savefig
        :return: None
            creates a .svg image into the path with the same name of input file
            If the data has more channels, creates more svg files.
        """
        out_name = path
        if path.endswith(".wav"):
            out_name = path[:-4]
        rate, data = wavfile.read(path)
        if data.ndim > 1:
            # we have more than one channel, we have to do more plots
            for i in range(0,data.ndim):
                dimension_data = data[:,i]
                out_path = out_name + "_channel_" + str(i) + "." + fmt
                fig = Spectrum.plot_spectrogram(rate, dimension_data);
                fig.savefig(out_path, format=fmt, frameon='false')
        else:
            out_name += "." + fmt
            fig = Spectrum.plot_spectrogram(rate, data);
            fig.savefig(out_name, format=fmt, frameon='false')
