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
            int_array = [int(numeric_string) for numeric_string in values]
            training_samples[ch].extend(int_array)
            
        pyplot.plot(training_samples['ch3'])
        pyplot.show()
        
        # Demean the channel
        demeaned_channel = training_samples['ch3'] - np.mean(training_samples['ch3'])
        
        freqs = np.fft.fftfreq(len(demeaned_channel), d = 1.0 / (EMG_SAMPLING_RATE))
        fft_powers = abs(np.fft.fft(demeaned_channel))
        
        pyplot.plot(freqs[:(freqs.size / 4)], fft_powers[:(freqs.size / 4)])
        pyplot.show()
            
    def classify_chunk(self):
        pass