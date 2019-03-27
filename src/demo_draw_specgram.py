import pickle
from Spectrum import Spectrum
import matplotlib.pyplot as plt

path = '/home/yanuba/Repos/WreckingNet/dataset/predict/ConcreteMixer_onsite.wav'

x = Spectrum.compute_specgram_and_delta(path)

fig, ax = plt.subplots(nrows = 2, ncols=1)
spectrum = x[:,:,0]
delta = x[:,:,1]

ax[0].imshow(spectrum, cmap='hot')
ax[0].set_title('spectrum')
ax[1].imshow(delta, cmap='hot')
ax[1].set_title('delta')
plt.show()
plt.close()
