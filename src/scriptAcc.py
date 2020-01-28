import numpy as np 
import matplotlib.pyplot as plt

import pickle
from Spectrum import Spectrum
import matplotlib.pyplot as plt

#---------------------------
'''# specgram

path = '/home/yanuba/Repos/WreckingNet/dataset/predict/ConcreteMixer_onsite.wav'
size = 440

x = Spectrum.compute_specgram_and_delta(path)

fig, ax = plt.subplots(nrows = 2, ncols=1)
spectrum = x[:,:size,0]
delta = x[:,:size,1]

print(spectrum.shape, delta.shape)

ax[0].imshow(spectrum, cmap='hot')
ax[0].set_title('spectrum')
ax[0].set_xlabel('Time bucket')
ax[0].set_ylabel('Mel band')

ax[1].imshow(delta, cmap='hot')
ax[1].set_title('delta')
ax[1].set_ylabel('Mel band')
ax[1].set_xlabel('Time bucket')

plt.show()
plt.close()'''

#-----------------------------
# sizes

sizeLabels = ["30", "50", "100", "500", "1000", "2000", "3000", "5000", "7000", "10000"]
accTestRaw = [94.76, 94.4, 95.8, 86.4, 91.4, 77.8, 62.6, 61.8, 39.4, 52]

fig,ax = plt.subplots()
#plot line
line, = plt.plot(sizeLabels,accTestRaw, marker="o")
#plot points

plt.plot(sizeLabels, accTestRaw, "ro")
plt.xlabel('Sample length (ms)')
plt.ylabel('Accuracy')
plt.suptitle('Test Accuracy')

annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    x,y = line.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = "{}, {}ms".format(" ".join([str(accTestRaw[n]) for n in ind["ind"]]), 
                           " ".join([sizeLabels[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = line.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
plt.close()

# accuracy over epochs

epochs = [str(x) for x in list(range(20))]
accuracy = [14.91, 91.31, 94.53, 95.29, 95.71, 96.13, 96.39, 96.50, 96.73, 96.86, 96.86, 96.85, 96.93, 96.94, 96.81, 96.97, 97.12, 96.76, 96.99, 97.04]

plt.plot(epochs, accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.suptitle('Accuracy history over 20 epochs')

plt.show()