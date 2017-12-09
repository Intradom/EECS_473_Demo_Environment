from time import sleep
import numpy as np
#from sklearn.cluster import KMeans
from matplotlib import pyplot
from matplotlib.pyplot import draw, figure, show
import sys
sys.path.append("../../head-sense-rpi3-software/")
#from sensor_packet_receiver import moving_avg_filter_t

"""

Left, under eye: Channel 0
Left, above eye: Channel 1
Right, above eye: Channel 2
Right, under eye: Channel 3

"""
 # 10 instances; right, under eye
TRAINING_SAMPLE_FILE_1 = "recorded_samples/scenario1.txt"
TRAINING_SAMPLE_FILE_2 = "recorded_samples/scenario2.txt"
TRAINING_SAMPLE_FILE_3 = "recorded_samples/scenario3.txt"
TRAINING_SAMPLE_FILE_4 = "recorded_samples/scenario4.txt"
TRAINING_SAMPLE_FILE_5 = "recorded_samples/scenario5.txt"
TRAINING_SAMPLE_FILE_6 = "recorded_samples/scenario6.txt"
TRAINING_SAMPLE_FILE_7 = "recorded_samples/scenario7.txt"
TRAINING_SAMPLE_FILE_8 = "recorded_samples/scenario8.txt"
TRAINING_SAMPLE_FILE_9 = "recorded_samples/scenario9.txt"
TRAINING_SAMPLE_FILE_10 = "recorded_samples/scenario10.txt"

#make a vector for all scenario files
files = [TRAINING_SAMPLE_FILE_1, TRAINING_SAMPLE_FILE_2, TRAINING_SAMPLE_FILE_3, TRAINING_SAMPLE_FILE_4, TRAINING_SAMPLE_FILE_5, TRAINING_SAMPLE_FILE_6, TRAINING_SAMPLE_FILE_7, TRAINING_SAMPLE_FILE_8, TRAINING_SAMPLE_FILE_9, TRAINING_SAMPLE_FILE_10]

# Parameters
EMG_SAMPLING_RATE = 1000
FFT_SIZE = 200
POSITIVE = 1
NEGATIVE = 0
SAMPLE_SIZE = 750
fileOfBlinks = open("training.csv")
row = []
csvFile = []

class EMG_Classifier:

    def __init__(self):
        
        master_file = []

        training_samples = dict()
        training_samples['ch0'] = list()
        training_samples['ch1'] = list()
        training_samples['ch2'] = list()
        training_samples['ch3'] = list()
        sample_index = 0
        
        #step 1
        for file in files:
            f_obj = open(file)
            # Skip first line
            for line in f_obj.readlines()[1:]:
                line = line.strip()
                [ch, values_str] = line.split(' :')
                values = values_str.split('[')[1].split(']')[0].split(', ')
                int_array = [int(numeric_string) for numeric_string in values]
                training_samples[ch].extend(int_array)

            master_file.append(training_samples)
    
        #step 2
        #parse csv file
        for line in fileOfBlinks.readlines():
            [file, channel, startTime] = line.split(',')
            row.append(int(file))
            row.append('ch' + channel)
            row.append(int(1000*float(startTime)))
            csvFile.append(row)
        
        segmented_data = []
        segmented_labels = []

        prev_s = -1
        prev_c = -1
        prev_seg_end = 0
        for line in range(len(csvFile)):
            # New section
            s = csvFile[line][0]
            c = csvFile[line][1]
            sp = csvFile[line][2] # starting point
            if (s != prev_s or c != prev_c):
                prev_seg_end = 0

            if (prev_seg_end > sp):
                # Error in training file, abort
                print("Error: " + str(s) + ", " + str(c) + ", " + str(sp))
                exit()

            # Get all negative samples before starting point
            for chunk in range(int((sp - prev_seg_end) / SAMPLE_SIZE)):
                seg_start = sp + chunk * SAMPLE_SIZE
                seg_end = seg_start + SAMPLE_SIZE
                segmented_data.append(master_file[s][c][seg_start:seg_end])
                segmented_labels.append(NEGATIVE)

            # Get 1 positive sample, ASSUMPTION: positive sample bounds will NOT go out of range of file
            prev_seg_end = sp + SAMPLE_SIZE
            segmented_data.append(master_file[s][c][sp:prev_seg_end])
            segmented_labels.append(POSITIVE)
        print (segmented_labels)
        
        
        ch_name = "ch0"
        detected_values = []
        ch_filter = moving_avg_filter_t(250, 250, -0.25, 500)

        f1 = figure()
        af1 = f1.add_subplot(111)
        f2 = figure()
        af2 = f2.add_subplot(111)
        af2.plot(training_samples[ch_name])
        show(block=False) 

        while sample_index < len(training_samples[ch_name]):
            samples = training_samples[ch_name][sample_index:sample_index+FFT_SIZE]
            #filtered_samples = []
            #for sample in samples:
            #    [diff_sample, fast_op, slow_op, detection_value] = ch_filter.signal_detect(sample)
            #    filtered_samples.append(slow_op)
            #samples = filtered_samples
            af1.plot(samples)
            pyplot.xlabel("Sample index = {0}".format(sample_index))
            pyplot.show()
        
            # Demean the channel
            demeaned_channel = samples - np.mean(samples)
        
            freqs = np.fft.fftfreq(len(demeaned_channel), d = 1.0 / (EMG_SAMPLING_RATE))
            fft_powers = abs(np.fft.fft(demeaned_channel))
        
            af1.plot(freqs[:int(freqs.size / 4)], fft_powers[:int(freqs.size / 4)])
            
            pyplot.xlabel("Sample index = {0}".format(sample_index))
            pyplot.show()

            sm = np.sum(fft_powers[0:5])/np.sum(fft_powers[0:10])
            if sm > 0.73:
                detected_values.append(1)
            else:
                detected_values.append(0)
            print("Mean for {0:4d} - {1:4d} => {2:.2f}".format(sample_index, sample_index + FFT_SIZE, sm))
            sample_index += FFT_SIZE
        af1.plot(detected_values)
        show()
        while True:
            sleep(1)
 
    def classify_chunk(self):
        pass
