from glottal_extracter import extract_glottal_signal
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import numpy as np

fileaudio="audio_5.wav"
fs, dataaudio=read(fileaudio)
if np.ndim(dataaudio) == 2:
	dataaudio = (dataaudio[:,0] + dataaudio[:,1])/2
print(dataaudio.shape)
glottal, g_iaif, GCIs=extract_glottal_signal(dataaudio, fs)

t = np.linspace(0, len(dataaudio)/fs, len(dataaudio))
plt.plot(t, glottal)
plt.xlabel('t')
plt.ylabel('Glottal Flow')
plt.show()