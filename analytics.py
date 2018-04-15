import pickle
import matplotlib.pyplot as plt
from scipy.signal import convolve
import numpy as np



with open('training_v2.pickle', 'rb') as handle:
    parsed_dict = pickle.load(handle)

# pickle load dictionary
print(parsed_dict.keys()) # labels
frequencies, times, spectogram, label = parsed_dict[list(parsed_dict.keys())[1]]
plt.pcolormesh(times, frequencies, spectogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

