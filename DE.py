#coding:utf-8
import math
import warnings
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def butter_bandpass_filter(data, lowcut, highcut, samplingRate, order=5):
	nyq = 0.5 * samplingRate
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	y = lfilter(b, a, data)
	return y

def compute_DE(data):
    variance = np.var(data, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def decompose(filepath,name):

    data = loadmat(filepath)
    frequency = 1000
    samples = data.shape[0]

    channels = data.shape[1]

    DE_Characteristics = np.empty([num_sample, channels, bands])

    temp_de = np.empty([0, num_sample])

    for channel in range(channels):

        trail_single = data[:, channel]

        Delta = butter_bandpass_filter(trail_single, 0.5, 4, frequency, order=3)
        Theta = butter_bandpass_filter(trail_single, 4, 8, frequency, order=3)
        Alpha = butter_bandpass_filter(trail_single, 8, 12, frequency, order=3)
        Beta = butter_bandpass_filter(trail_single, 12, 30, frequency, order=3)
        Gamma = butter_bandpass_filter(trail_single, 30, 50, frequency, order=3)

        for index in range(num_sample):
            DE_Delta = np.append(DE_Delta, compute_DE(Delta[index * 100: (index + 1) * 100]))
            DE_Theta = np.append(DE_Theta, compute_DE(Theta[index * 100: (index + 1) * 100]))
            DE_alpha = np.append(DE_alpha, compute_DE(Alpha[index * 100: (index + 1) * 100]))
            DE_beta = np.append(DE_beta, compute_DE(Beta[index * 100: (index + 1) * 100]))
            DE_gamma = np.append(DE_gamma, compute_DE(Gamma[index * 100: (index + 1) * 100]))

        temp_de = np.vstack([temp_de, DE_Delta])
        temp_de = np.vstack([temp_de, DE_Theta])
        temp_de = np.vstack([temp_de, DE_alpha])
        temp_de = np.vstack([temp_de, DE_beta])
        temp_de = np.vstack([temp_de, DE_gamma])

    temp_trail_de = temp_de.reshape(-1, 5, num_sample)
    print("trail_DE shape", DE_Characteristics.shape)
    temp_trail_de = temp_trail_de.transpose([2, 0, 1])
    DE_Characteristics = np.vstack([temp_trail_de])

    print("trail_DE shape", DE_Characteristics.shape)
    return DE_Characteristics

filepath = 'D:/Program Files/'

RawdataName = ['yw.mat']

EEGName = ['EEG01']

for i in range(len(RawdataName)):
    dataFile = filepath + RawdataName[i]
    print('processing {}'.format(RawdataName[i]))
    DE_Characteristics = decompose(dataFile, EEGName[i])
    DE = np.vstack([DE, DE_Characteristics])

np.save("D:/Work_2/MyDataset/yw.npy", DE)