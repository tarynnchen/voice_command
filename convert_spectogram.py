import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import glob
import pickle
import wavio


def list_directory(dir):
    folders = []
    for root, dirs, files in os.walk(dir):
        for name in dirs:
            folders.append(os.path.join(root, name))
    return folders

def parse_pandas(folders):
    parsed_dict = {}
    for folder in folders:
        counter = 0
        for file in glob.glob(os.path.join(os.path.join(data_dir, folder), '*.wav')):
            sample_rate, samples = wavfile.read(os.path.join(folder, file))
            frequencies, times, spectogram = signal.spectrogram(samples,sample_rate)
            parsed_dict[folder.split('/')[-1]+str(counter)] = [frequencies, times, spectogram,folder.split('/')[-1]]
            counter += 1
    return parsed_dict

data_dir = '/Users/weiyuchen/Documents/Project/VoiceCommand/Dataset'
folders = list_directory(data_dir)
parsed_dict = parse_pandas([folders[-2],folders[-8]])

'''
sample_rate, samples = wavfile.read('/Users/weiyuchen/Documents/Project/VoiceCommand/Dataset/bed/0a7c2a8d_nohash_0.wav')
frequencies, times, spectogram = signal.spectrogram(samples, sample_rate) # sample_rate is defined as frequency (Hz)

plt.pcolormesh(times, frequencies, spectogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
'''
with open('training_v2.pickle', 'wb') as handle:
    pickle.dump(parsed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


