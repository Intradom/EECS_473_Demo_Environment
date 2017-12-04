import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot

"""

Left, under eye: Channel 0
Left, above eye: Channel 1
Right, above eye: Channel 2
Right, under eye: Channel 3

"""
 # 10 instances; right, under eye
TRAINING_SAMPLE_FILE = "recorded_samples/scenario1.txt"

# Parameters
EMG_SAMPLING_RATE = 1000

class EMG_Classifier:

    def __init__(self):
        f_obj = open(TRAINING_SAMPLE_FILE)
        
        training_samples = dict()
        training_samples['ch0'] = list()
        training_samples['ch1'] = list()
        training_samples['ch2'] = list()
        training_samples['ch3'] = list()
        
        # Skip first line
        for line in f_obj.readlines()[1:]:
            line = line.strip()
            [ch, values_str] = line.split(' :')
            values = values_str.split("[")[1].split(']')[0].split(', ')
            training_samples[ch].extend(values)
            
        pyplot.plot(training_samples['ch3'])
        pyplot.show()
        
        freqs = np.fft.fftfreq(len(training_samples['ch3']), d = 1.0 / (EMG_SAMPLING_RATE))
        fft_powers = np.fft.fft(training_samples['ch3'])
        
        pyplot.plot(freqs, fft_powers)
        pyplot.show()
            
    def classify_chunk(self):
        pass